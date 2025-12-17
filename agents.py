import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from abc import ABC, abstractmethod

from networks import *


# --------------------------------------------------------------- #
#                     Base Agent Class Definition                 #
# --------------------------------------------------------------- #
class BaseAgent(ABC):
    """Abstract base class for all agents with common functionality."""
    
    def __init__(self, cfg):
        # Environment setup (may create vectorized envs)
        self.device = cfg.get("device", torch.device("cpu"))
        self._setup_environment(cfg)

        # Setup state and action space dimensions
        if hasattr(self.env, "single_observation_space"):
            self.state_dim = self.env.single_observation_space.shape[0]
        else:
            self.state_dim = self.env.observation_space.shape[0]

        if hasattr(self.env, "single_action_space"):
            self._setup_action_dim_parallel(cfg)
        else:
            self._setup_action_dim(cfg)
        
        # Setup common hyperparameters
        self.gamma = cfg["gamma"]
        self.render_int = cfg.get("display_episodes", 100)
        self.num_episodes = cfg.get("num_episodes", cfg.get("episodes", 1000))
        self.max_steps = cfg.get("max_steps", 200)
        
        # Exploration parameters
        # let each one decide if they want to set up these parameters
        # self._setup_exploration_params(cfg)

    def _setup_environment(self, cfg):
        """Setup environment based on config."""
        # for parallel environments, create a synchronous VectorEnv
        if cfg.get("parallel_envs", False):
            self.num_envs = cfg.get("num_envs", 2)

            # no render for vectorized envs
            # self.env = gym.vector.SyncVectorEnv(env_fns)
            self.env = gym.make_vec(
                cfg["env_name"],
                num_envs=self.num_envs,
                render_mode=None,
                vectorization_mode="sync"
            )
            # do not use human render with vector envs
            self.render_env = None
            self.display = False

        else:
            self.env = gym.make(cfg["env_name"], render_mode=None)
            self.render_env = gym.make(cfg["env_name"], render_mode="human")
            self.display = cfg.get("display", False)
            self.num_envs = 1

    def _setup_action_dim_parallel(self, cfg):
        """Setup action dimension for vectorized environments."""
        try:
            self.action_space_type = cfg["action_space_type"]
            if self.action_space_type == "continuous":
                self.action_dim = self.env.single_action_space.shape[0]
            else:
                self.action_dim = self.env.single_action_space.n

        except KeyError:
            raise ValueError("action_space_type not specified in config. Please specify 'discrete' or 'continuous'.")


    def _setup_action_dim(self, cfg):
        """Setup action dimension based on action space type."""
        try:
            self.action_space_type = cfg["action_space_type"]
            if self.action_space_type == "continuous":
                self.action_dim = self.env.action_space.shape[0]
            else:
                self.action_dim = self.env.action_space.n

        except KeyError:
            raise ValueError("action_space_type not specified in config. Please specify 'discrete' or 'continuous'.")
    
    def _setup_exploration_params(self, cfg):
        """Setup epsilon-greedy or Boltzmann exploration parameters."""
        self.use_boltzmann = cfg.get(
            f"{self._get_agent_prefix()}_use_boltzmann", False)
        prefix = self._get_agent_prefix()
        
        try:
            # Epsilon-greedy parameters
            self.epsilon = cfg[f"{prefix}_initial_epsilon"]
            self.min_epsilon = cfg[f"{prefix}_min_epsilon"]
            self.decay = cfg[f"{prefix}_decay_rate"]
            
            # Boltzmann parameters
            self.tau = cfg[f"{prefix}_initial_tau"]
            self.min_tau = cfg[f"{prefix}_min_tau"]
            self.decay_rate_tau = cfg[f"{prefix}_decay_rate_tau"]

        except KeyError:
            raise ValueError(f"Missing exploration parameters for agent type: {prefix}")

        if self.use_boltzmann:
            self.exp_strat = "boltzmann"
        else:
            self.exp_strat = "epsilon"

    def _setup_entropy_params(self, cfg):
        self.exp_strat = "entropy_reg"
        self.use_entropy_regularization = cfg["use_entropy_reg"]
        self.beta = cfg["beta"]
    
    @abstractmethod
    def _get_agent_prefix(self):
        """Return the config prefix for this agent type (e.g., 'sarsa', 'reinforce')."""
        pass
    
    def decay_exploration_param(self):
        """Decay epsilon or tau."""
        if self.use_boltzmann:
            self.tau = max(self.min_tau, self.tau * np.exp(-self.decay_rate_tau))
        else:
            self.epsilon = max(self.min_epsilon, self.epsilon * np.exp(-self.decay))
    
    def get_env(self, episode):
        """Get the appropriate environment (render or non-render) for the episode."""
        return self.render_env if self.display and (episode + 1) % self.render_int == 0 else self.env
    
    def print_progress(self, episode, total_reward, rewards_history):
        """Print training progress."""
        if (episode + 1) % self.render_int == 0:
            avg_reward = np.mean(rewards_history[-20:])

            if self.exp_strat == "epsilon":
                exploration_param = f"ε={self.epsilon:.3f}"
            elif self.exp_strat == "boltzmann":
                exploration_param = f"τ={self.tau:.3f}"
            elif self.exp_strat == "entropy_reg":
                exploration_param = f"β={self.beta:.3f}"
            else:
                exploration_param = ""

            print(f"Episode {episode+1}/{self.num_episodes} | Reward {total_reward:.1f} | "
                  f"Avg {avg_reward:.1f} | {exploration_param}", flush=True)
    
    @abstractmethod
    def select_action(self, state):
        """Select an action given the current state."""
        pass
    
    @abstractmethod
    def train(self):
        """Main training loop."""
        pass


class SarsaAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._setup_exploration_params(cfg)
        
        # Policy network, optimizer, and loss function
        self.q_net = ValueNet(
            self.state_dim, self.action_dim,
            cfg["hidden_layers"], cfg["activation_function"]).to(self.device)
        self.opt = optim.Adam(self.q_net.parameters(), lr=cfg["learning_rate"])
        self.criterion = nn.MSELoss()
    
    def _get_agent_prefix(self):
        return "sarsa"

    def select_action(self, state: torch.Tensor):
        """
        Epsilon-greedy action selection.

        Args:
            state (np.array): Current state.
        Returns:
            action (int): Selected action.
        """
        # get Q-values from the network without tracking gradients
        # gradients are generated in the training loop
        with torch.no_grad():
            q_values = self.q_net(state)

        if self.use_boltzmann:
            prob_dist = F.softmax(q_values / self.tau, dim=-1)
            action = torch.multinomial(prob_dist, 1).item()

        else:
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_dim)

            else:
                action = torch.argmax(q_values).item()

        return action

    def train(self):
        """SARSA training loop implementation."""
        print(f"Starting training with parameters gamma={self.gamma}, use_boltzmann={self.use_boltzmann}...")
        rewards_all = []

        for m in range(self.num_episodes):
            total = 0
            env = self.get_env(m)
            
            # ==== EXPERIENCE COLLECTION PHASE ====
            experiences = []
            init_obs = env.reset()
            s = init_obs[0]
            a = self.select_action(torch.Tensor(s).to(self.device))

            # Gather experiences for the episode
            for t in range(self.max_steps):
                obs = env.step(a)
                s_next, r, done = obs[0], obs[1], obs[2]
                a_next = self.select_action(torch.Tensor(s_next).to(self.device))

                experiences.append((s, a, r, s_next, a_next, done))

                s = s_next
                a = a_next
                total += r
                
                if self.display and (m + 1) % self.render_int == 0:
                    env.render()

                if done:
                    break

            # ==== BATCH UPDATE PHASE ====
            states = torch.FloatTensor(
                np.array([exp[0] for exp in experiences])).to(self.device)
            actions = torch.LongTensor(
                np.array([exp[1] for exp in experiences])).to(self.device)
            rewards = torch.FloatTensor(
                np.array([exp[2] for exp in experiences])).to(self.device)
            next_states = torch.FloatTensor(
                np.array([exp[3] for exp in experiences])).to(self.device)
            next_actions = torch.LongTensor(
                np.array([exp[4] for exp in experiences])).to(self.device)
            dones = torch.FloatTensor(
                np.array([exp[5] for exp in experiences])).to(self.device)

            # Compute Q-values and targets
            q_values = self.q_net(states)
            pred_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = self.q_net(next_states)
                next_q_for_actions = next_q_values.gather(
                    1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + self.gamma * next_q_for_actions * (1 - dones)

            # Update network
            loss = self.criterion(pred_q_values, targets)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            rewards_all.append(total)
            self.print_progress(m, total, rewards_all)
            self.decay_exploration_param()

        return rewards_all


class ReinforceAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._setup_exploration_params(cfg)
        
        self.use_baseline = cfg.get("reinforce_use_baseline", False)
        self.action_space_type = cfg.get("action_space_type", "discrete")
        
        if self.action_space_type == "continuous":
            self.policy = ContinuousPolicyNet(
                self.state_dim,
                self.action_dim,
                cfg["hidden_layers"],
                cfg["activation_function"]
            ).to(self.device)
        else:
            self.policy = PolicyNet(
                self.state_dim,
                self.action_dim,
                cfg["hidden_layers"],
                cfg["activation_function"]
            ).to(self.device)
        
        self.opt = optim.Adam(self.policy.parameters(), lr=cfg["learning_rate"])

        if self.use_baseline:
            self.baseline = nn.Linear(self.state_dim, 1).to(self.device)
            self.baseline_opt = optim.Adam(self.baseline.parameters(), lr=cfg["learning_rate"])
    
    def _get_agent_prefix(self):
        return "reinforce"

    def select_action(self, state: np.array):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if self.action_space_type == "continuous":
            # Continuous action space using Gaussian policy
            mean, log_std = self.policy(state_t)
            mean = mean.squeeze()
            log_std = log_std.squeeze()
            std = torch.exp(log_std)
            
            # Add temperature/exploration scaling
            if self.use_boltzmann:
                std = std * self.tau
            
            # Sample action from Gaussian distribution
            dist = torch.distributions.Normal(mean, std)
            action_t = dist.sample()
            action_t = torch.clamp(action_t, -1, 1)
            
            # Compute log probability by summing over action dimensions
            log_prob = dist.log_prob(action_t).sum()
            
            return action_t.cpu().numpy(), log_prob
        
        else:  # Discrete action space
            logits_t, probs_t = self.policy(state_t)
            logits_t = logits_t.squeeze()
            probs_t = probs_t.squeeze()

            if self.use_boltzmann:
                # Numerically stable softmax with temperature
                scaled_logits = logits_t / self.tau
                boltz_probs = torch.softmax(scaled_logits, dim=-1)
                action = torch.multinomial(boltz_probs, 1).item()
                log_prob = torch.log(torch.clamp(boltz_probs[action], min=1e-8))
                
                return action, log_prob

            else:
                # Epsilon-greedy
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.action_dim)
                else:
                    # create a categorical distribution over the actions
                    dist = torch.distributions.Categorical(probs=probs_t)
                    # sample an action from the distribution
                    action = dist.sample()
                    action = action.item()

            log_prob = torch.log(torch.clamp(probs_t[action], min=1e-8))
            return action, log_prob

    def update(self, log_probs, rewards, states: np.array):
        R, returns = 0, []

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        if self.use_baseline:
            states_t = torch.FloatTensor(states).to(self.device)
            baselines = self.baseline(states_t).squeeze()
            advantages = returns - baselines.detach()
            base_loss = F.mse_loss(baselines, returns)
            self.baseline_opt.zero_grad()
            base_loss.backward()
            self.baseline_opt.step()
        else:
            advantages = returns
        
        # maximize expected return = minimize -expected return
        loss = -torch.sum(torch.stack([lp * adv for lp, adv in zip(log_probs, advantages)]))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self):
        """REINFORCE training loop implementation."""
        print(f"Starting training with parameters gamma={self.gamma}, use_boltzmann={self.use_boltzmann}...")
        rewards_all = []

        for ep in range(self.num_episodes):
            env = self.get_env(ep)
            state, _ = env.reset()
            log_probs, rewards, states = [], [], []
            total = 0

            for t in range(self.max_steps):
                action, log_prob = self.select_action(state)
                log_probs.append(log_prob)
                
                next_state, reward, done, trunc, _ = env.step(action)
                
                rewards.append(reward)
                states.append(state)
                total += reward
                state = next_state

                if (ep + 1) % self.render_int == 0 and self.display:
                    env.render()

                if done or trunc:
                    break

            # convert to numpy array for better performance
            states = np.array(states)
            self.update(log_probs, rewards, states)
            self.decay_exploration_param()
            rewards_all.append(total)
            self.print_progress(ep, total, rewards_all)

        return rewards_all


class A2CAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._setup_entropy_params(cfg)

        if self.action_space_type == "continuous":
            self.actor = ContinuousPolicyNet(
                self.state_dim,
                self.action_dim,
                cfg["actor_hidden_layers"],
                cfg["activation_function"]
            ).to(self.device)
        else:
            self.actor = PolicyNet(
                self.state_dim,
                self.action_dim,
                cfg["actor_hidden_layers"],
                cfg["activation_function"]
            ).to(self.device)

        # critic is producing values (V(s)), so it only needs to produce one output value
        self.critic = ValueNet(
            self.state_dim,
            1,
            cfg["critic_hidden_layers"],
            cfg["activation_function"]
        ).to(self.device)

        # Optimizers
        self.actor_opt = optim.Adam(
            self.actor.parameters(), lr=cfg["learning_rate"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg["learning_rate"])
        
        # Criterion
        self.criterion = nn.MSELoss()
        
        # Hyperparameters
        self.use_gae = cfg["use_gae"]
        self.lambda_ = cfg["lambda"]
        self.N = cfg["n_steps"]

        if self.use_gae:
            self.approx_func = self.gae
        else:
            self.approx_func = self.N_step_return
    
    def _get_agent_prefix(self):
        return "a2c"


    def select_action(self, state: np.array):
        """Sample action from actor network."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if self.action_space_type == "continuous":
            mean, log_std = self.actor(state_t)
            mean = mean.squeeze()
            log_std = log_std.squeeze()
            std = torch.exp(log_std)
            
            dist = torch.distributions.Normal(mean, std)
            action_t = dist.sample()
            action_t = torch.clamp(action_t, -1, 1)
            log_prob = dist.log_prob(action_t).sum()
            entropy = dist.entropy().sum()
            
            return action_t.cpu().numpy(), log_prob, entropy
        
        else:  # Discrete
            _, probs_t = self.actor(state_t)
            probs_t = probs_t.squeeze()
            
            dist = torch.distributions.Categorical(probs=probs_t)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device))
            entropy = dist.entropy().sum()
            
            return action, log_prob, entropy


    def N_step_return(self, rewards_t, dones_t, Vs, N):
        """
        Computes the advantages and target values using the
        N-step approximation for Q(s, a).

        All computations are done with no_grad() to avoid unnecessary gradient computation.

        Args:
            rewards_t (torch.Tensor): list of rewards for each state-action pair.
            dones_t (torch.Tensor): list of done flags for each state-action pair.
            Vs (torch.Tensor): list of state values for each state.
            N (int): number of steps in the trajectory.
        Returns:
            advantages (torch.Tensor): list of advantages for each state-action pair.
            targets (torch.Tensor): list of target values for each state-action pair.
        """
        with torch.no_grad():
            rets = torch.zeros(N, device=self.device)
            future_ret = Vs[-1] * (1 - dones_t[-1])

            for t in reversed(range(N)):
                rets[t] = rewards_t[t] + self.gamma * \
                    future_ret * (1 - dones_t[t])
                future_ret = rets[t]

            # Compute advantages: A(s,a) = Q(s,a) - V(s) ≈ Return - V(s)
            advantages = rets - Vs
            # set target values
            targets = rets

            return advantages, targets

    def gae(self, rewards_t, dones_t, Vs, N):
        """
        This function computes the advantages and target values using the
        Generalized Advantage Estimation (GAE).

        All computations are done with no_grad() to avoid unnecessary gradient computation.
        
        Args:
            rewards_t (torch.Tensor): list of rewards for each state-action pair.
            dones_t (torch.Tensor): list of done flags for each state-action pair.
            Vs (torch.Tensor): list of state values for each state.
            N (int): number of steps in the trajectory.
        Returns:
            advantages (torch.Tensor): list of advantages for each state-action pair.
            targets (torch.Tensor): list of target values for each state-action pair.
        """
        with torch.no_grad():
            # gaes = advantages
            gaes = torch.zeros_like(rewards_t, device=self.device)
            future_gae = torch.tensor(0.0, dtype=rewards_t.dtype, device=self.device)

            for t in reversed(range(N)):
                # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
                delta = rewards_t[t] + self.gamma * \
                    Vs[t + 1] * (1 - dones_t[t]) - Vs[t]

                # GAE: A_t = δ_t + (γλ)*δ_{t+1} + (γλ)^2*δ_{t+2} + ...
                gaes[t] = delta + self.gamma * self.lambda_ * (1 - dones_t[t]) * future_gae
                future_gae = gaes[t]

            # Target is V(s) + A(s,a)
            targets = Vs[:-1] + gaes
            return gaes, targets

    def update(self, log_probs, entropies, rewards, states, dones):
        """
        Update both the actor and the critic networks.

        Args:
            log_probs (list of torch.Tensor): list of log probabilities for each action.
            rewards (list of float): list of rewards for each state-action pair.
            states (list of np.array): list of states for each state-action pair.
            dones (list of bool): list of done flags for each state-action pair.
        """
        N = len(rewards)
        states_t = torch.FloatTensor(states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Ensure the correct shapes
        assert len(states_t) == N + 1
        assert len(rewards_t) == N
        assert len(dones_t) == N
        assert len(log_probs) == N
        
        # Compute all state values at once
        Vs = self.critic(states_t).squeeze()

        # Compute advantages and targets
        # pass Vs without gradient information
        advantages, targets = self.approx_func(
            rewards_t, dones_t, Vs, N
        )
        
        # Update critic to minimize TD error        
        critic_loss = self.criterion(Vs[:-1], targets)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Update actor to maximize expected advantage
        actor_loss = -torch.sum(torch.stack(log_probs) * advantages.detach())

        if self.use_entropy_regularization:
            # Add entropy regularization term
            entropies_t = torch.stack(entropies).to(self.device)
            actor_loss += -self.beta * entropies_t.sum()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

    def train(self):
        print(
            f"Starting training for A2C...")
        rewards_all = []

        for ep in range(self.num_episodes):
            env = self.get_env(ep)
            state, _ = env.reset()
            log_probs, entropies = [], []
            rewards, states, dones = [], [], []
            total = 0

            states.append(state)
            
            for t in range(self.max_steps):
                action, log_prob, entropy = self.select_action(state)
                next_state, reward, done, trunc, _ = env.step(action)

                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(reward)
                states.append(next_state)
                dones.append(done or trunc)
            
                total += reward
                state = next_state

                if (ep + 1) % self.render_int == 0 and self.display:
                    env.render()

                if done or trunc:
                    break

            # convert to numpy array for better performance
            states = np.array(states)
            self.update(log_probs, entropies, rewards, states, dones)
            # TODO: Add entropy regularization call here
            # self.decay_exploration_param()
            rewards_all.append(total)
            self.print_progress(ep, total, rewards_all)

        return rewards_all


    def train_parallel(self):
        """Train using synchronous vectorized environments.

        Collects `N`-step rollouts for each parallel environment, concatenates
        per-environment trajectories (so each env's transitions are contiguous),
        and calls `update()` with the flattened lists.
        Falls back to `train()` if not running with multiple envs.
        """
        # Fallback to regular train if not vectorized
        if getattr(self, "num_envs", 1) <= 1:
            return self.train()

        env = self.env
        rewards_all = []

        rollout_len = min(self.N, self.max_steps)

        for ep in range(self.num_episodes):
            # reset returns (obs, info) for gymnasium
            obs, _ = env.reset()
            # obs shape: (num_envs, state_dim)

            # Per-env buffers
            num_envs = self.num_envs
            states_bufs = [[obs[i]] for i in range(num_envs)]
            rewards_bufs = [[] for _ in range(num_envs)]
            dones_bufs = [[] for _ in range(num_envs)]
            logp_bufs = [[] for _ in range(num_envs)]
            ent_bufs = [[] for _ in range(num_envs)]
            total_rewards = np.zeros(num_envs, dtype=float)

            # Collect rollout_len steps
            for t in range(rollout_len):
                # select actions for each env (do not vectorize actor here)
                actions = []
                for i in range(num_envs):
                    a, lp, ent = self.select_action(states_bufs[i][-1])
                    actions.append(a)
                    logp_bufs[i].append(lp)
                    ent_bufs[i].append(ent)

                # prepare action array for vector env
                if self.action_space_type == "continuous":
                    actions_arr = np.stack(actions)
                else:
                    actions_arr = np.array(actions)

                next_obs, rewards, terminations, truncations, infos = env.step(actions_arr)

                for i in range(num_envs):
                    rewards_bufs[i].append(rewards[i])
                    dones_bufs[i].append(bool(terminations[i] or truncations[i]))
                    states_bufs[i].append(next_obs[i])
                    total_rewards[i] += rewards[i]

            # Flatten per-env buffers into global lists with env-major ordering
            flat_states = []
            flat_rewards = []
            flat_dones = []
            flat_logp = []
            flat_ent = []

            for i in range(num_envs):
                # each states_bufs[i] is length rollout_len+1
                flat_states.extend(states_bufs[i])
                flat_rewards.extend(rewards_bufs[i])
                flat_dones.extend(dones_bufs[i])
                flat_logp.extend(logp_bufs[i])
                flat_ent.extend(ent_bufs[i])

            # convert states to numpy array shape (N_total+1, state_dim)
            flat_states = np.array(flat_states)

            # call update with flattened sequences
            self.update(flat_logp, flat_ent, flat_rewards, flat_states, flat_dones)

            # record mean reward across envs for this rollout
            avg_total = float(np.mean(total_rewards))
            rewards_all.append(avg_total)
            self.print_progress(ep, avg_total, rewards_all)

        return rewards_all


class PPOAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _get_agent_prefix(self):
        return "ppo"
    
    def select_action(self, state):
        pass

    def update(self, trajectories):
        pass
    
    def train(self):
        pass
