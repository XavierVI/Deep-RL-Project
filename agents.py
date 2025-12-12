import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from abc import ABC, abstractmethod

# --------------------------------------------------------------- #
#                     Policy Class Definitions                    #
# --------------------------------------------------------------- #
class SarsaQNet(nn.Module):
    def __init__(self, s_dim, a_dim, hidden, act):
        super().__init__()

        act_fn = getattr(F, act)
        self.layers = nn.ModuleList()
        prev = s_dim

        for h in hidden:
            self.layers.append(nn.Linear(prev, h))
            prev = h

        self.out = nn.Linear(prev, a_dim)
        self.act_fn = act_fn

    def forward(self, x):
        for l in self.layers:
            x = self.act_fn(l(x))

        return self.out(x)


class PolicyNet(nn.Module):
    """Policy network for discrete action spaces."""
    def __init__(self, s_dim, a_dim, hidden, act):
        super().__init__()

        # this dynamically gets the activation function from torch.nn.functional
        act_fn = getattr(F, act)
        print(f"Using activation function: {act}")
        self.layers = nn.ModuleList()
        prev = s_dim
        
        for h in hidden:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        
        self.out = nn.Linear(prev, a_dim)
        self.act_fn = act_fn
    
    def forward(self, x):
        for l in self.layers:
            x = self.act_fn(l(x))

        x = self.out(x)

        # return logits and action probabilities
        return x, F.softmax(x, dim=-1)


class ContinuousPolicyNet(nn.Module):
    """Policy network for continuous action spaces using Gaussian policy."""
    def __init__(self, s_dim, a_dim, hidden, act):
        super().__init__()

        act_fn = getattr(F, act)
        print(f"Using activation function: {act}")
        self.layers = nn.ModuleList()
        prev = s_dim
        
        for h in hidden:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        
        # Output mean and log_std for Gaussian policy
        self.mean = nn.Linear(prev, a_dim)
        self.log_std = nn.Linear(prev, a_dim)
        self.act_fn = act_fn
    
    def forward(self, x):
        for l in self.layers:
            x = self.act_fn(l(x))

        mean = torch.tanh(self.mean(x))  # Bound mean to [-1, 1]
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent numerical instability
        
        return mean, log_std



# --------------------------------------------------------------- #
#                     Agent Class Definitions                     #
# --------------------------------------------------------------- #
class BaseAgent(ABC):
    """Abstract base class for all agents with common functionality."""
    
    def __init__(self, cfg):
        # Environment setup
        self.env = gym.make(cfg["env_name"], render_mode=None)
        self.render_env = gym.make(cfg["env_name"], render_mode="human")
        self.display = cfg.get("display", False)
        self.device = cfg.get("device", torch.device("cpu"))
        
        # State and action dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self._setup_action_dim(cfg)
        
        # Common hyperparameters
        self.gamma = cfg["gamma"]
        self.render_int = cfg.get("display_episodes", 100)
        self.num_episodes = cfg.get("num_episodes", cfg.get("episodes", 1000))
        self.max_steps = cfg.get("max_steps", 200)
        
        # Exploration parameters
        self.use_boltzmann = cfg.get(f"{self._get_agent_prefix()}_use_boltzmann", False)
        self._setup_exploration_params(cfg)
        
    def _setup_action_dim(self, cfg):
        """Setup action dimension based on action space type."""
        try:
            action_space_type = cfg["action_space_type"]
            if action_space_type == "continuous":
                self.action_dim = self.env.action_space.shape[0]
            else:
                self.action_dim = self.env.action_space.n

        except KeyError:
            raise ValueError("action_space_type not specified in config. Please specify 'discrete' or 'continuous'.")
    
    def _setup_exploration_params(self, cfg):
        """Setup epsilon-greedy or Boltzmann exploration parameters."""
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
    
    @abstractmethod
    def _get_agent_prefix(self):
        """Return the config prefix for this agent type (e.g., 'sarsa', 'reinforce')."""
        pass
    
    def decay_epsilon(self):
        """Decay epsilon or tau for exploration."""
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
            avg_reward = np.mean(rewards_history[-20:]) if rewards_history else 0
            exploration_param = f"τ={self.tau:.3f}" if self.use_boltzmann else f"ε={self.epsilon:.3f}"
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
        
        # Policy network, optimizer, and loss function
        self.q_net = SarsaQNet(
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
            self.decay_epsilon()

        return rewards_all


class ReinforceAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        
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
            self.decay_epsilon()
            rewards_all.append(total)
            self.print_progress(ep, total, rewards_all)

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
