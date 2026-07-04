# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/pqn/#pqn_atari_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import json


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

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 20000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches (FIXED)"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.001
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""

    # GNS Specific Arguments
    use_gns: bool = True
    """Toggle dynamic rollout size adjustment using GNS"""
    gns_update_freq: int = 50
    """How often (in iterations) to update the rollout size based on GNS."""
    num_steps_min: int = 16
    """Minimum number of steps per rollout"""
    num_steps_max: int = 64
    """Maximum number of steps per rollout"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.LayerNorm([32, 20, 20]),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LayerNorm([64, 9, 9]),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LayerNorm([64, 7, 7]),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, env.single_action_space.n)),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


#######################################################################################################
#######################################################################################################
# GNS Calculation Helper (Corrected per McCandlish et al. 2018)
def estimate_gns(q_network, b_obs, b_actions, b_returns, batch_size, num_micro_batches=8):
    """
    Estimates Gradient Noise Scale (GNS) using the definition:
    B_noise = tr(Sigma) / |G|^2
    
    We estimate this by splitting the batch into smaller micro-batches.
    """
    # Ensure we can split evenly or handle the remainder
    micro_batch_size = batch_size // num_micro_batches
    if micro_batch_size < 1:
        return 0.0

    grads = []
    
    # Compute gradients for each micro-batch
    for i in range(num_micro_batches):
        start = i * micro_batch_size
        end = start + micro_batch_size
        
        # Slice data
        mb_obs = b_obs[start:end]
        mb_actions = b_actions[start:end]
        mb_returns = b_returns[start:end]
        
        # Forward pass
        old_val = q_network(mb_obs).gather(1, mb_actions.unsqueeze(-1).long()).squeeze()
        loss = F.mse_loss(mb_returns, old_val)
        
        # Compute gradients
        g = torch.autograd.grad(loss, q_network.parameters(), create_graph=False)
        
        # Flatten and store
        grads.append(torch.cat([x.flatten() for x in g]))

    # Stack gradients: [num_micro_batches, num_params]
    grads = torch.stack(grads)
    
    # 1. Compute Mean Gradient (g_avg)
    g_avg = grads.mean(dim=0)
    
    # 2. Compute Variance (Noise)
    diff = grads - g_avg
    sq_diff = diff.pow(2).sum(dim=1)
    
    # Unbiased estimator of variance of micro-batches
    var_micro = sq_diff.sum() / (num_micro_batches - 1)
    
    # tr(Sigma) approx var_micro * micro_batch_size
    noise_measure = var_micro * micro_batch_size
    
    # 3. Compute Signal (|G|^2)
    g_avg_sq = g_avg.norm().pow(2)
    signal_measure = g_avg_sq - (noise_measure / batch_size)
    
    # Handle numerical instability
    if signal_measure <= 1e-8:
        return 0.0
        
    # 4. Calculate GNS
    gns = noise_measure / signal_measure
    
    return gns.item()
#######################################################################################################
#######################################################################################################


def evaluate_agent(q_network, num_episodes=100, num_eval_envs=10):
    eval_env = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=num_eval_envs,
        episodic_life=False,
        reward_clip=False,
        seed=args.seed + 100,
    )
    
    episode_returns = []
    episode_lengths = []
    
    current_returns = np.zeros(num_eval_envs)
    current_lengths = np.zeros(num_eval_envs)
    
    q_network.eval()
    
    obs = torch.Tensor(eval_env.reset()).to(device)
    completed_episodes = 0
    
    with torch.no_grad():
        while completed_episodes < num_episodes:
            q_values = q_network(obs)
            actions = torch.argmax(q_values, dim=1)
            
            next_obs, rewards, dones, infos = eval_env.step(actions.cpu().numpy())
            
            current_returns += infos["reward"]
            current_lengths += 1
            
            for idx in range(num_eval_envs):
                if infos["terminated"][idx]:
                    if completed_episodes < num_episodes:
                        episode_returns.append(current_returns[idx])
                        episode_lengths.append(current_lengths[idx])
                        completed_episodes += 1
                        
                        current_returns[idx] = 0
                        current_lengths[idx] = 0
            
            obs = torch.Tensor(next_obs).to(device)
    
    eval_env.close()
    q_network.train()
    
    return episode_returns, episode_lengths


if __name__ == "__main__":

    args = tyro.cli(Args)
    
    # Initial batch size calculation
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    # Store initial values for reference
    initial_num_steps = args.num_steps
    initial_batch_size = args.batch_size
    
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"

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
    writer = SummaryWriter(f"runs/PQN/{args.env_id}/{run_name}")
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
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.RAdam(q_network.parameters(), lr=args.learning_rate)

    # ALGO Logic: Storage setup - Use maximum possible size
    max_storage_size = args.num_steps_max
    obs = torch.zeros((max_storage_size, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((max_storage_size, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((max_storage_size, args.num_envs)).to(device)
    dones = torch.zeros((max_storage_size, args.num_envs)).to(device)
    values = torch.zeros((max_storage_size, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    iteration = 0
    while global_step < args.total_timesteps:
        iteration += 1
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (global_step / args.total_timesteps)
            base_lr = frac * args.learning_rate
        else:
            base_lr = args.learning_rate
        
        optimizer.param_groups[0]["lr"] = base_lr

        # Rollout phase - collect data for current num_steps
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,)).to(device)
            with torch.no_grad():
                q_values = q_network(next_obs)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()

            explore = torch.rand((args.num_envs,)).to(device) < epsilon
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    print(f"[{args.env_id}] global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # Compute Q(lambda) targets - only for collected steps
        with torch.no_grad():
            returns = torch.zeros_like(rewards[:args.num_steps]).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_value, _ = torch.max(q_network(next_obs), dim=-1)
                    nextnonterminal = 1.0 - next_done
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                    returns[t] = (
                        rewards[t]
                        + args.gamma * (args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value) * nextnonterminal
                    )

        # Flatten the batch - only collected data
        b_obs = obs[:args.num_steps].reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions[:args.num_steps].reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        # Optimizing the Q-network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                old_val = q_network(b_obs[mb_inds]).gather(1, b_actions[mb_inds].unsqueeze(-1).long()).squeeze()
                loss = F.mse_loss(b_returns[mb_inds], old_val)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("losses/td_loss", loss, global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        #######################################################################################################
        # GNS Logic: Estimate and Adjust Rollout Size AFTER Q update
        if args.use_gns and iteration % args.gns_update_freq == 0:
            # Estimate GNS using the current batch
            current_gns = estimate_gns(q_network, b_obs, b_actions, b_returns, args.batch_size, num_micro_batches=8)
            
            # Determine new num_steps based on GNS
            # GNS represents the optimal batch size in terms of gradient noise
            # We scale num_steps to achieve target batch size (GNS)
            # target_batch_size = GNS
            # num_steps = target_batch_size / num_envs
            target_batch_size = int(current_gns)
            new_num_steps = max(args.num_steps_min, min(target_batch_size // args.num_envs, args.num_steps_max))
            
            # Update args for next iteration
            args.num_steps = new_num_steps
            args.batch_size = args.num_envs * args.num_steps
            args.minibatch_size = args.batch_size // args.num_minibatches
            
            writer.add_scalar("charts/GNS", current_gns, global_step)
            writer.add_scalar("charts/num_steps", args.num_steps, global_step)
            writer.add_scalar("charts/batch_size", args.batch_size, global_step)
            writer.add_scalar("charts/minibatch_size", args.minibatch_size, global_step)
            
            print(f"GNS Update: GNS={current_gns:.2f}, num_steps={args.num_steps}, batch_size={args.batch_size}")
        #######################################################################################################

    envs.close()
    
    print("\n" + "="*50)
    print(f"Starting evaluation for {args.env_id} with 100 episodes...")
    print("="*50 + "\n")
    
    eval_start_time = time.time()
    episode_returns, episode_lengths = evaluate_agent(
        q_network, 
        num_episodes=100,
        num_eval_envs=100
    )
    eval_time = time.time() - eval_start_time
    
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    min_return = np.min(episode_returns)
    max_return = np.max(episode_returns)
    median_return = np.median(episode_returns)
    mean_length = np.mean(episode_lengths)
    
    print("Overall Evaluation Results:")
    print("="*50)
    print(f"Number of Episodes: {len(episode_returns)}")
    print(f"Evaluation Time: {eval_time:.2f}s")
    print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Median Return: {median_return:.2f}")
    print(f"Min Return: {min_return:.2f}")
    print(f"Max Return: {max_return:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print("="*50 + "\n")
    
    writer.add_scalar("evaluation/mean_return", mean_return, global_step)
    writer.add_scalar("evaluation/std_return", std_return, global_step)
    writer.add_scalar("evaluation/median_return", median_return, global_step)
    writer.add_scalar("evaluation/min_return", min_return, global_step)
    writer.add_scalar("evaluation/max_return", max_return, global_step)
    writer.add_scalar("evaluation/mean_length", mean_length, global_step)
    writer.add_scalar("evaluation/eval_time", eval_time, global_step)
    
    eval_results = {
        'num_episodes': len(episode_returns),
        'num_eval_envs': 100,
        'eval_time': eval_time,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'mean_return': mean_return,
        'std_return': std_return,
        'median_return': median_return,
        'min_return': min_return,
        'max_return': max_return,
        'mean_length': mean_length,
    }
    
    results_path = f"runs/PQN/{args.env_id}/{run_name}/eval_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"Evaluation results saved to: {results_path}")
    
    model_path = f"runs/PQN/{args.env_id}/{run_name}/q_network.pth"
    torch.save({
        'model_state_dict': q_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'args': vars(args),
        'eval_results': eval_results,
    }, model_path)
    print(f"Model saved to: {model_path}")

    writer.close()
