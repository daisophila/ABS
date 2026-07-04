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
    env_id: str = "Amidar-v5"
    """the id of the environment"""
    total_timesteps: int = 20000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
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
    
    # Network architecture arguments
    use_multiskip: bool = False
    """whether to use MultiSkipResidualMLP"""
    mlp_hidden_size: int = 512
    """hidden size for MLP"""
    mlp_num_layers: int = 5 # large: 5,  xlarge: 10
    """number of residual layers in MLP"""
    use_ln: bool = True
    """whether to use layer normalization"""
    activation_fn: str = "relu"
    """activation function (relu, gelu, elu, etc.)"""
    cnn_channels: tuple = (64, 128, 128)
    """CNN channel sizes"""
    
    # Adaptive rollout arguments
    adapt_rollout: bool = False
    """whether to adapt num_steps based on policy change"""
    num_steps_min: int = 16
    """minimum number of rollout steps"""
    num_steps_max: int = 64
    """maximum number of rollout steps"""
    rollout_adapt_freq: int = 50
    """how often to adapt rollout size (in iterations)"""
    policy_change_high: float = 0.95
    """policy change threshold for small rollout"""
    policy_change_low: float = 0.05
    """policy change threshold for large rollout"""
    batch_sch_type: str = 'log'
    """linear, log, exp"""
    
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


def get_act_fn_clss(activation_fn):
    """Get activation function class from string"""
    activation_fn = activation_fn.lower()
    if activation_fn == "relu":
        return nn.ReLU
    elif activation_fn == "gelu":
        return nn.GELU
    elif activation_fn == "elu":
        return nn.ELU
    elif activation_fn == "tanh":
        return nn.Tanh
    elif activation_fn == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise ValueError(f"Unknown activation function: {activation_fn}")


###################### AtariCNN ######################
class AtariCNN(nn.Module):
    def __init__(
        self,
        cnn_channels,
        use_ln=False,
        activation_fn="relu",
        kernel_sizes=[8, 4, 3],
        ln_sizes=[20, 9, 7],
        strides=[4, 2, 1],
        in_channels=4,
        input_size=84,
        device='cpu'
    ):
        super().__init__()
        act_ = get_act_fn_clss(activation_fn)
        cnn = []
        for out_channel, kernel_size, stride, ln_size in zip(cnn_channels, kernel_sizes, strides, ln_sizes):
            cnn.append(
                layer_init(
                    nn.Conv2d(in_channels, out_channel, kernel_size, stride=stride, device=device),
                )
            )
                        
            if use_ln:
                cnn.append(
                    nn.LayerNorm(
                        [out_channel, ln_size, ln_size], device=device
                    )
                )
                
            cnn.append(
                act_()
            )
            
            output_size = (input_size - kernel_size) / stride + 1
            input_size = output_size
            in_channels = out_channel
            
        cnn.append(
            nn.Flatten()
        )
        self.cnn = nn.Sequential(*cnn)
        
        self.output_size = int(out_channel * output_size * output_size)
        
    def forward(self, x):
        return self.cnn(x)


###################### MultiSkip Residual MLP ######################
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, use_ln=False, use_spectral_norm=False, activation_fn="relu", device='cpu', linear_clss=nn.Linear):
        super().__init__()
        act_fn_class = get_act_fn_clss(activation_fn)
        
        if linear_clss.__name__ == 'NoisyLinear':
            layer1 = linear_clss(hidden_size, hidden_size, device=device)
            layer2 = linear_clss(hidden_size, hidden_size, device=device)
        else:
            layer1 = layer_init(linear_clss(hidden_size, hidden_size, device=device))
            layer2 = layer_init(linear_clss(hidden_size, hidden_size, device=device))
        
        if use_spectral_norm:
            from torch.nn.utils import spectral_norm
            layer1 = spectral_norm(layer1)
            layer2 = spectral_norm(layer2)
            
        layers = [layer1]
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size, device=device))
        layers.append(act_fn_class())
        
        layers.append(layer2)
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size, device=device))
        
        self.block = nn.Sequential(*layers)
        self.act = act_fn_class()
        
    def forward(self, x):
        return self.act(x + self.block(x))


class StoreGlobalSkip(nn.Module):
    def forward(self, x):
        return (x, x)


class MultiSkipResidualBlock(nn.Module):
    def __init__(self, hidden_size, use_ln=False, use_spectral_norm=False, activation_fn="relu", device='cpu', linear_clss=nn.Linear):
        super().__init__()
        self.block = ResidualBlock(hidden_size, use_ln=use_ln, use_spectral_norm=use_spectral_norm, activation_fn=activation_fn, device=device, linear_clss=linear_clss)
        
    def forward(self, x_tuple):
        x, global_skip = x_tuple
        out = self.block(x)
        out = out + global_skip
        return (out, global_skip)


class ExtractOutput(nn.Module):
    def forward(self, x_tuple):
        x, _ = x_tuple
        return x


class MultiSkipResidualMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        last_act=False,
        use_ln=False,
        use_spectral_norm=False,
        activation_fn="relu",
        device='cpu',
        linear_clss=nn.Linear
    ):
        super().__init__()
        self.output_size = output_size
        act_fn_class = get_act_fn_clss(activation_fn)
        layers = []
        
        if linear_clss.__name__ == 'NoisyLinear':
            layer = linear_clss(input_size, hidden_size, device=device)
        else:
            layer = layer_init(linear_clss(input_size, hidden_size, device=device))
        
        if use_spectral_norm:
            from torch.nn.utils import spectral_norm
            layer = spectral_norm(layer)
        layers.append(layer)
            
        if use_ln:
            layers.append(nn.LayerNorm(hidden_size, device=device))
        
        layers.append(StoreGlobalSkip())
        
        for _ in range(num_layers):
            layers.append(MultiSkipResidualBlock(hidden_size, use_ln=use_ln, use_spectral_norm=use_spectral_norm, activation_fn=activation_fn, device=device, linear_clss=linear_clss))
        
        layers.append(ExtractOutput())
        
        if linear_clss.__name__ == 'NoisyLinear':
            layer = linear_clss(hidden_size, output_size, device=device)
        else:
            layer = layer_init(linear_clss(hidden_size, output_size, device=device))
        
        if use_spectral_norm:
            from torch.nn.utils import spectral_norm
            layer = spectral_norm(layer)
        layers.append(layer)
            
        if use_ln:
            layers.append(nn.LayerNorm(output_size, device=device))
        if last_act:
            layers.append(act_fn_class())
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


###################### Q-Network ######################
class QNetwork_Multi_Skip(nn.Module):
    def __init__(self, env, args, device='cpu'):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = AtariCNN(
            cnn_channels=args.cnn_channels,
            use_ln=args.use_ln,
            activation_fn=args.activation_fn,
            device=device
        )
        
        # MLP head
        self.mlp = MultiSkipResidualMLP(
            input_size=self.cnn.output_size,
            hidden_size=args.mlp_hidden_size,
            output_size=env.single_action_space.n,
            num_layers=args.mlp_num_layers,
            last_act=False,
            use_ln=args.use_ln,
            activation_fn=args.activation_fn,
            device=device
        )

    def forward(self, x):
        x = self.cnn(x / 255.0)
        return self.mlp(x)


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


#######################################################################################################
#######################################################################################################
class AdaptiveRolloutScheduler:
    def __init__(self, args, envs, device):
        self.args = args
        self.envs = envs
        self.device = device
        self.old_q_network = None
        self.policy_changes = deque(maxlen=20)
        
    def should_adapt(self, iteration):
        return self.args.adapt_rollout and iteration % self.args.rollout_adapt_freq == 0
    
    @torch.no_grad()
    def measure_policy_change(self, q_network, obs_buffer):
        total_obs = obs_buffer.reshape(-1, *obs_buffer.shape[2:])
        sample_size = min(2048, len(total_obs))
        indices = np.random.choice(len(total_obs), sample_size, replace=False)
        sample_obs = total_obs[indices]
        
        old_actions = torch.argmax(self.old_q_network(sample_obs), dim=-1)
        new_actions = torch.argmax(q_network(sample_obs), dim=-1)
        change_rate = (old_actions != new_actions).float().mean().item()
        
        self.policy_changes.append(change_rate)
        
        return change_rate

    @torch.no_grad()
    def update_policy(self, q_network):
        if self.old_q_network is None:
            self.old_q_network = QNetwork(self.envs, self.args, self.device).to(self.device)
            self.old_q_network.load_state_dict(q_network.state_dict())
            self.old_q_network.eval()
        else:        
            self.old_q_network.load_state_dict(q_network.state_dict())
    
    def adapt_num_steps(self, policy_change):
        if len(self.policy_changes) < 10:
            return self.args.num_steps_min

        avg_change = np.mean(self.policy_changes) if len(self.policy_changes) > 5 else policy_change
        
        change_min = self.args.policy_change_low
        change_max = self.args.policy_change_high

        log_change = np.log(np.clip(avg_change, change_min, change_max))
        log_min = np.log(change_min)
        log_max = np.log(change_max)
        
        normalized = (log_change - log_min) / (log_max - log_min)
        
        target_steps = int(self.args.num_steps_max - normalized * (self.args.num_steps_max - self.args.num_steps_min))
        target_steps = max(self.args.num_steps_min, min(self.args.num_steps_max, target_steps))

        new_steps = int(0.9 * self.args.num_steps + 0.1 * target_steps)
        new_steps = max(self.args.num_steps_min, min(self.args.num_steps_max, new_steps))

        self.args.num_steps = new_steps

        return new_steps
    
    def reallocate_storage(self):
        obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_action_space.shape).to(self.device)
        rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        
        return obs, actions, rewards, dones, values
#######################################################################################################
#######################################################################################################


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def evaluate_agent(q_network, args, device, num_episodes=100, num_eval_envs=10):
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
                        
                        print(f"Evaluation Episode {completed_episodes}/{num_episodes}: "
                              f"Return = {current_returns[idx]:.2f}, "
                              f"Length = {current_lengths[idx]}")
                        
                        current_returns[idx] = 0
                        current_lengths[idx] = 0
            
            obs = torch.Tensor(next_obs).to(device)
    
    eval_env.close()
    q_network.train()
    
    return episode_returns, episode_lengths


if __name__ == "__main__":

    args = tyro.cli(Args)
    
    if args.adapt_rollout:
        args.num_steps = args.num_steps_min
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.update_epochs = max(2*args.num_steps // 32, 2)
    
    args.num_iterations = args.total_timesteps // args.batch_size + 1000
    
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

    if args.use_multiskip:
        q_network = QNetwork_Multi_Skip(envs, args, device).to(device)
    else:
        q_network = QNetwork(envs).to(device)
    optimizer = optim.RAdam(q_network.parameters(), lr=args.learning_rate)

    # Adaptive rollout scheduler
    scheduler = AdaptiveRolloutScheduler(args, envs, device)
    
    # ALGO Logic: Storage setup
    obs, actions, rewards, dones, values = scheduler.reallocate_storage()
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    iteration = 0
    while global_step < args.total_timesteps:
        iteration += 1
        if scheduler.should_adapt(iteration):
            scheduler.update_policy(q_network)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - global_step / args.total_timesteps
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout
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

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)


        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(device)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
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

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("losses/td_loss", loss, global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        
        sps = int(global_step / (time.time() - start_time))
        print(f"Iter {iteration}, Steps {global_step}/{args.total_timesteps}, SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        if scheduler.should_adapt(iteration):
            policy_change = scheduler.measure_policy_change(q_network, obs)
            
            old_num_steps = args.num_steps
            args.num_steps = scheduler.adapt_num_steps(policy_change)
            
            if args.num_steps != old_num_steps:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}: Adapting rollout size")
                print(f"  num_steps: {old_num_steps} → {args.num_steps}")
                print(f"  policy_change: {policy_change:.4f}")
                print(f"  avg_policy_change: {np.mean(scheduler.policy_changes):.4f}")
                print(f"{'='*60}\n")
                
                obs, actions, rewards, dones, values = scheduler.reallocate_storage()
                
                args.batch_size = args.num_steps * args.num_envs
                args.minibatch_size = args.batch_size // args.num_minibatches
                args.update_epochs = max(2*args.num_steps // 32, 2)

            writer.add_scalar("adapt/policy_change", policy_change, global_step)
            writer.add_scalar("adapt/avg_policy_change", np.mean(scheduler.policy_changes), global_step)
            writer.add_scalar("adapt/num_steps", args.num_steps, global_step)
            writer.add_scalar("adapt/batch_size", args.batch_size, global_step)
            writer.add_scalar("adapt/minibatch_size", args.minibatch_size, global_step)

    envs.close()

    print("\n" + "="*50)
    print("Starting evaluation with 100 episodes...")
    print("="*50 + "\n")
    
    eval_start_time = time.time()
    episode_returns, episode_lengths = evaluate_agent(
        q_network,
        args,
        device,
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
    
    print("\nOverall Evaluation Results:")
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
    
    import json
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
