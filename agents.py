import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from abc import ABC, abstractmethod

from networks import *
from EnvironmentManager import EnvironmentManager
from Exploration import ExplorationStrategy

# --------------------------------------------------------------- #
#                     Base Agent Class Definition                 #
# --------------------------------------------------------------- #
class BaseAgent(ABC):
    """Abstract base class for all agents with common functionality."""
    
    def __init__(self, cfg):
        # Environment setup (may create vectorized envs)
        self.device = cfg.get("device", torch.device("cpu"))
        self.env = EnvironmentManager(cfg)

        # Setup state and action space dimensions
        self.state_dim = self.env.get_state_space_size()
        self.action_dim = self.env.get_action_space_size()
        
        # Setup common hyperparameters
        self.gamma = cfg["gamma"]
        self.render_int = cfg.get("display_episodes", 100)
        self.num_episodes = cfg.get("num_episodes", cfg.get("episodes", 1000))
        self.max_steps = cfg.get("max_steps", 200)
        
        # Exploration parameters
        self.exp_strat = ExplorationStrategy(cfg)
    
    @abstractmethod
    def _get_agent_prefix(self):
        """Return the config prefix for this agent type (e.g., 'sarsa', 'reinforce')."""
        pass
    
    def print_progress(self, episode, total_reward, rewards_history):
        """Print training progress."""
        if (episode + 1) % self.render_int == 0:
            avg_reward = np.mean(rewards_history[-20:])
            exploration_param = self.exp_strat.get_exploration_param()

            if self.exp_strat.get_strategy() == "epsilon":
                exploration_param = f"ε={exploration_param:.3f}"
            elif self.exp_strat.get_strategy() == "boltzmann":
                exploration_param = f"τ={exploration_param:.3f}"
            elif self.exp_strat.get_strategy() == "entropy_reg":
                exploration_param = f"β={exploration_param:.3f}"
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
        self.use_boltzmann = self.exp_strat.get_strategy() == "boltzmann"
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
            tau = self.exp_strat.get_exploration_param()
            prob_dist = F.softmax(q_values / tau, dim=-1)
            action = torch.multinomial(prob_dist, 1).item()

        else:
            epsilon = self.exp_strat.get_exploration_param()
            if np.random.rand() < epsilon:
                action = np.random.randint(self.action_dim)

            else:
                action = torch.argmax(q_values).item()

        return action

    def train(self):
        """SARSA training loop implementation."""
        print(f"Starting training with SARSA...")
        rewards_all = []

        for m in range(self.num_episodes):
            total = 0
            env = self.env.get_env(m)
            
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
                
                self.env.render(m)

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
            self.exp_strat.decay_exploration_param()

        return rewards_all


class ReinforceAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.use_baseline = cfg.get("reinforce_use_baseline", False)
        self.action_space_type = cfg.get("action_space_type", "discrete")

        self.use_boltzmann = self.exp_strat.get_strategy() == "boltzmann"
        
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
                tau = self.exp_strat.get_exploration_param()
                std = std * tau
            
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
                tau = self.exp_strat.get_exploration_param()
                # Numerically stable softmax with temperature
                scaled_logits = logits_t / tau
                boltz_probs = torch.softmax(scaled_logits, dim=-1)
                action = torch.multinomial(boltz_probs, 1).item()
                log_prob = torch.log(torch.clamp(boltz_probs[action], min=1e-8))
                
                return action, log_prob

            else:
                epsilon = self.exp_strat.get_exploration_param()
                # Epsilon-greedy
                if np.random.rand() < epsilon:
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
        print(f"Starting training with REINFORCE...")
        rewards_all = []

        for ep in range(self.num_episodes):
            env = self.env.get_env(ep)
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

                self.env.render(ep)

                if done or trunc:
                    break

            # convert to numpy array for better performance
            states = np.array(states)
            self.update(log_probs, rewards, states)
            self.exp_strat.decay_exploration_param()
            rewards_all.append(total)
            self.print_progress(ep, total, rewards_all)

        return rewards_all


class A2CAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        if self.env.action_space_type == "continuous":
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
        self.criterion = nn.MSELoss(reduction='mean')
        
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

    def N_step_return(self, rewards_t, dones_t, Vs, T):
        """
        Computes the advantages and target values using the
        N-step approximation for Q(s, a).

        All computations are done with no_grad() to avoid unnecessary gradient computation.

        Args:
            rewards_t (torch.Tensor): list of rewards for each state-action pair.
            dones_t (torch.Tensor): list of done flags for each state-action pair.
            Vs (torch.Tensor): list of state values for each state.
            T (int): number of steps in the trajectory.
        Returns:
            advantages (torch.Tensor): list of advantages for each state-action pair.
            targets (torch.Tensor): list of target values for each state-action pair.
        """
        with torch.no_grad():
            rets = torch.zeros(T, self.env.num_envs, device=self.device)
            future_ret = Vs[-1] * (1 - dones_t[-1])

            for t in reversed(range(min(self.N, T))):
                rets[t] = rewards_t[t] + self.gamma * \
                    future_ret * (1 - dones_t[t])
                future_ret = rets[t]

            # Compute advantages: A(s,a) = Q(s,a) - V(s) ≈ Return - V(s)
            advantages = rets - Vs
            # set target values
            targets = rets

            return advantages, targets

    def gae(self, rewards_t, dones_t, Vs, T):
        """
        This function computes the advantages and target values using the
        Generalized Advantage Estimation (GAE).

        All computations are done with no_grad() to avoid unnecessary gradient computation.
        
        Args:
            rewards_t (torch.Tensor): list of rewards for each state-action pair.
            dones_t (torch.Tensor): list of done flags for each state-action pair.
            Vs (torch.Tensor): list of state values for each state.
            T (int): number of steps in the trajectory.
        Returns:
            advantages (torch.Tensor): list of advantages for each state-action pair.
            targets (torch.Tensor): list of target values for each state-action pair.
        """
        with torch.no_grad():
            # gaes = advantages
            gaes = torch.zeros(T, self.env.num_envs, device=self.device)
            future_gae = torch.zeros(self.env.num_envs, device=self.device)

            # if N > T, use T steps
            for t in reversed(range(T)):
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
        T = len(rewards)
        states_t = torch.FloatTensor(states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Ensure the correct shapes
        assert len(states_t) == T + 1
        assert len(rewards_t) == T
        assert len(dones_t) == T
        assert len(log_probs) == T
        
        # Compute all state values at once
        Vs = self.critic(states_t).squeeze(-1)

        # Compute advantages and targets
        # pass Vs without gradient information
        advantages, targets = self.approx_func(
            rewards_t, dones_t, Vs, T
        )
        
        # Update critic to minimize TD error        
        critic_loss = self.criterion(Vs[:-1], targets)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Update actor to maximize expected advantage
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()

        if self.exp_strat.get_strategy() == "entropy_reg":
            # Add entropy regularization term
            beta = self.exp_strat.get_exploration_param()
            entropies_t = torch.stack(entropies).to(self.device)
            actor_loss += -beta * entropies_t.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

    def select_action(self, state: np.array):
        """Sample action from actor network."""
        if self.env.num_envs == 1:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_t = torch.FloatTensor(state).to(self.device)

        if self.env.action_space_type == "continuous":
            mean, log_std = self.actor(state_t)
            mean = mean.squeeze()
            log_std = log_std.squeeze()
            std = torch.exp(log_std)

            dist = torch.distributions.Normal(mean, std)
            action_t = dist.sample()
            action_t = torch.clamp(action_t, -1, 1)
            log_prob_t = dist.log_prob(action_t).sum(dim=-1)
            entropy_t = dist.entropy().sum(dim=-1)

        else:  # Discrete
            # here, we only need the probabilities
            _, probs_t = self.actor(state_t)

            dist = torch.distributions.Categorical(probs=probs_t)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t).sum(dim=-1)
            entropy_t = dist.entropy().sum(dim=-1)

        # we need to save the log prob and entropy grads
        # for the update step
        return action_t.cpu().numpy(), log_prob_t, entropy_t

    def train(self):
        print(f"Starting training for A2C...")
        rewards_all = []

        for ep in range(self.num_episodes):
            env = self.env.get_env(ep)
            state, _ = env.reset()
            log_probs, entropies = [], []
            rewards, states, dones = [], [], []
            total = 0

            states.append(state)
            
            for t in range(self.max_steps):
                actions, log_probs_t, entropies_t = self.select_action(state)
                next_state, reward, done, trunc, _ = env.step(actions)

                log_probs.append(log_probs_t)
                entropies.append(entropies_t)
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
