import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from tensorboard.compat.proto.types_pb2 import DT_DOUBLE
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ABS"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments (fixed to 1)"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    num_eval_episodes: int = 10
    
    # Adaptive rollout arguments
    adapt_rollout: bool = True
    """whether to adapt num_steps based on KL divergence"""
    num_steps_min: int = 1024
    """minimum number of rollout steps"""
    num_steps_max: int = 8192
    """maximum number of rollout steps"""
    rollout_adapt_freq: int = 10
    """how often to adapt rollout size (in iterations)"""
    kl_high: float = 0.1
    """KL threshold for small rollout (high policy change)"""
    kl_low: float = 0.01
    """KL threshold for large rollout (low policy change)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""



def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def get_action(self, x):
        return self.actor_mean(x)


#######################################################################################################
#######################################################################################################
class AdaptiveRolloutScheduler:
    def __init__(self, args, envs, device):
        self.args = args
        self.envs = envs
        self.device = device
        self.kl_history = deque(maxlen=20)
        self.return_history = deque(maxlen=100)

        self.old_agent = None
        
    def should_adapt(self, iteration):
        return self.args.adapt_rollout and iteration % self.args.rollout_adapt_freq == 0
    
    @torch.no_grad()
    def compute_kl_divergence(self, agent, obs_buffer):

        total_obs = obs_buffer.reshape(-1, *obs_buffer.shape[2:])
        
        sample_size = min(2048, len(total_obs))
        indices = np.random.choice(len(total_obs), sample_size, replace=False)
        sample_obs = total_obs[indices]

        current_mean = agent.actor_mean(sample_obs)
        current_logstd = agent.actor_logstd.expand_as(current_mean)
        current_std = torch.exp(current_logstd)

        old_mean = self.old_agent.actor_mean(sample_obs)
        old_logstd = self.old_agent.actor_logstd.expand_as(old_mean)
        old_std = torch.exp(old_logstd)
        
        kl_div = torch.log(old_std / current_std) + \
                 (current_std.pow(2) + (current_mean - old_mean).pow(2)) / (2 * old_std.pow(2)) - 0.5
        
        kl_div = kl_div.sum(dim=-1).mean()
        
        kl_value = kl_div.item()
        
        
        return kl_value
    
    def update_pi_old(self, agent):
        if self.old_agent is None:
            self.old_agent = Agent(self.envs).to(self.device)
            self.old_agent.load_state_dict(agent.state_dict())
            self.old_agent.eval()
        else:
            self.old_agent.load_state_dict(agent.state_dict())
    
    def adapt_num_steps(self, kl_div):
        
        self.kl_history.append(kl_div)
        
        if len(self.kl_history) < 10:
            avg_kl = self.args.kl_high
        else:
            avg_kl = np.mean(self.kl_history) if len(self.kl_history) > 5 else kl_div
        ess, ess_ratio = 0., 0.

        change_min = self.args.kl_low
        change_max = self.args.kl_high
        
        log_change = np.log(np.clip(avg_kl, change_min, change_max))
        log_min = np.log(change_min)
        log_max = np.log(change_max)
        
        normalized = (log_change - log_min) / (log_max - log_min)
        
        target_steps = int(self.args.num_steps_max - normalized * (self.args.num_steps_max - self.args.num_steps_min))
        target_steps = max(self.args.num_steps_min, min(self.args.num_steps_max, target_steps))
        
        new_steps = int(0.9 * self.args.num_steps + 0.1 * target_steps)
        new_steps = max(self.args.num_steps_min, min(self.args.num_steps_max, new_steps))
        new_steps = (new_steps // self.args.num_minibatches) * self.args.num_minibatches
        if new_steps < self.args.num_steps_min:
            new_steps = self.args.num_steps_min
        
        return new_steps, avg_kl, ess, ess_ratio
    
    def reallocate_storage(self):
        obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        
        return obs, actions, logprobs, rewards, dones, values

#######################################################################################################
#######################################################################################################


if __name__ == "__main__":

    args = tyro.cli(Args)
    
    if args.adapt_rollout:
        args.num_steps = args.num_steps_min
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    args.num_iterations = args.total_timesteps // args.batch_size + 1000
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/PPO/{run_name}_{args.num_steps_min}-{args.num_steps_max}_kl{args.kl_high}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    #######################################################################################################
    # Adaptive rollout scheduler
    scheduler = AdaptiveRolloutScheduler(args, envs, device)
    #######################################################################################################

    # ALGO Logic: Storage setup
    obs, actions, logprobs, rewards, dones, values = scheduler.reallocate_storage()

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    iteration = 0
    while global_step < args.total_timesteps:
        iteration += 1
        
        if scheduler.should_adapt(iteration):
            scheduler.update_pi_old(agent)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - global_step / args.total_timesteps
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout Collection
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={float(info['episode']['r']):.2f}")
                        scheduler.return_history.append(info['episode']['r'])
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        approx_kl_current = None
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kl_current = approx_kl.item()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        #######################################################################################################
        if scheduler.should_adapt(iteration):
            kl_div = scheduler.compute_kl_divergence(agent, obs)
            
            old_num_steps = args.num_steps
            new_num_steps, avg_kl, ess, ess_ratio = scheduler.adapt_num_steps(kl_div)
            args.num_steps = new_num_steps
            
            if args.num_steps != old_num_steps:
                print(f"\n{'='*70}")
                print(f"Iteration {iteration}: Adapting rollout size")
                print(f"  num_steps: {old_num_steps} → {args.num_steps}")
                print(f"  KL(cur||old): {kl_div:.6f}  (policy update magnitude)")
                print(f"  avg KL (last 20): {avg_kl:.6f}")
                print(f"  ESS: {ess:.1f} / {old_num_steps} (ratio: {ess_ratio:.3f})")
                if len(scheduler.return_history) > 10:
                    recent_return_mean = np.mean(list(scheduler.return_history)[-10:])
                    print(f"  recent avg return: {recent_return_mean:.2f}")
                print(f"{'='*70}\n")

                obs, actions, logprobs, rewards, dones, values = scheduler.reallocate_storage()

                args.batch_size = args.num_steps * args.num_envs
                args.minibatch_size = args.batch_size // args.num_minibatches
                args.update_epochs = 10*args.num_steps // 2048
                
            writer.add_scalar("adapt/kl_cur_old", kl_div, global_step)
            writer.add_scalar("adapt/avg_kl", avg_kl, global_step)
            writer.add_scalar("adapt/num_steps", args.num_steps, global_step)
            writer.add_scalar("adapt/batch_size", args.batch_size, global_step)
            writer.add_scalar("adapt/ess", ess, global_step)
            writer.add_scalar("adapt/ess_ratio", ess_ratio, global_step)
        #######################################################################################################

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        sps = int(global_step / (time.time() - start_time))
        print(f"Iter {iteration}, Steps {global_step}/{args.total_timesteps}, SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.ABS_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
