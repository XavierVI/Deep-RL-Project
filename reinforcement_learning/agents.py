import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from abc import ABC, abstractmethod

from reinforcement_learning.networks import *
from reinforcement_learning.EnvironmentManager import EnvironmentManager
from reinforcement_learning.Exploration import ExplorationStrategy

from reinforcement_learning.base_agent import BaseAgent


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

        self._init_actor_critic(cfg)
        
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

    def _check_parameter_shapes(self, T, states_t, rewards_t, dones_t, log_probs_t):
        """Ensure the correct shapes of the tensors."""
        assert len(states_t) == T + 1
        assert len(rewards_t) == T
        assert len(dones_t) == T
        assert len(log_probs_t) == T

    def _compute_advantages(self, rewards_t, dones_t, Vs, T):
        # pass Vs without gradient information
        advantages, targets = self.approx_func(
            rewards_t, dones_t, Vs, T
        )

        # normalize advantages
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        return advantages, targets

    def _update_critic(self, Vs, targets):
        critic_loss = self.criterion(Vs[:-1], targets)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

    def _update_actor(self, log_probs, advantages, entropies):
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()

        if self.exp_strat.get_strategy() == "entropy_reg":
            # Add entropy regularization term
            beta = self.exp_strat.get_exploration_param()
            entropies_t = torch.stack(entropies).to(self.device)
            actor_loss += -beta * entropies_t.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

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
        # entropies_t.shape == log_probs_t.shape
        self._check_parameter_shapes(T, states_t, rewards_t, dones_t, log_probs)
        
        # Compute all state values at once
        Vs = self.critic(states_t).view(T + 1, self.env.num_envs)

        # Compute advantages and targets
        advantages, targets = self.compute_advantages(rewards_t, dones_t, Vs, T)

        # Update critic to minimize TD error        
        self._update_critic(Vs, targets)

        # Update actor to maximize expected advantage
        self._update_actor(log_probs, advantages, entropies)

    def select_action(self, state: np.array):
        """Sample action from actor network."""
        if self.env.num_envs == 1:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_t = torch.FloatTensor(state).to(self.device)

        if self.env.action_space_type == "continuous":
            mean, log_std = self.actor(state_t)
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
            log_prob_t = dist.log_prob(action_t)
            entropy_t = dist.entropy()

        # we need to save the log prob and entropy grads
        # for the update step
        return action_t.cpu().numpy(), log_prob_t, entropy_t

    def train(self, profiler=None):
        print(f"Starting training for A2C...")
        rewards_all = []

        for ep in range(self.num_episodes):
            env = self.env.get_env(ep)
            state, _ = env.reset()
            log_probs, entropies = [], []
            rewards, states, dones = [], [], []
            total: float = 0.0

            states.append(state)
            
            # collect rollout
            for t in range(self.max_steps):
                actions, log_probs_t, entropies_t = self.select_action(state)
                next_state, reward, done, trunc, _ = env.step(actions)

                log_probs.append(log_probs_t)
                entropies.append(entropies_t)
                rewards.append(reward)
                states.append(next_state)
                dones.append(np.array(done) | np.array(trunc))

                state = next_state

                self.env.render(ep)

                # if done, reset the environment and continue
                # collecting data
                if self.env.num_envs == 1:
                    total += reward
                    if done or trunc:
                        state, _ = env.reset()
                else:
                    total += np.mean(reward)
                    if any(done) or any(trunc):
                        state, _ = env.reset()

            # convert to numpy array for better performance
            states = np.array(states)
            rewards = np.array(rewards)
            dones = np.array(dones)
            self.update(log_probs, entropies, rewards, states, dones)
            # TODO: Add entropy regularization call here
            # self.decay_exploration_param()
            rewards_all.append(total)
            self.print_progress(ep, total, rewards_all)

        return rewards_all

class PPOAgent(A2CAgent):
    """
    Proximal Policy Optimization (PPO) Agent implementation.

    Only uses GAE for advantage estimation.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # override advantage function to always use GAE
        self.approx_func = self.gae

        # Criterion
        self.criterion = nn.MSELoss(reduction='mean')
        self.minibatch_size = cfg.get("ppo_minibatch_size", 64)
    
    def _get_agent_prefix(self):
        return "ppo"
    
    def select_action(self, state):
        pass

    def _update_actor(self, log_probs, advantages, entropies):
        """
        PPO-specific actor update using clipped surrogate objective.
        """

        
    
    def update(self, old_log_probs, entropies, rewards, states, dones):
        """
        Update both the actor and the critic networks.

        Args:
            old_log_probs (list of torch.Tensor): list of log probabilities for each action.
            rewards (list of float): list of rewards for each state-action pair.
            states (list of np.array): list of states for each state-action pair.
            dones (list of bool): list of done flags for each state-action pair.
        """
        T = len(rewards)
        states_t = torch.FloatTensor(states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Ensure the correct shapes
        assert len(states_t) == T + 1
        assert len(rewards_t) == T
        assert len(dones_t) == T
        assert len(old_log_probs_t) == T

        # Compute all state values at once
        Vs = self.critic(states_t).view(T + 1, self.env.num_envs)

        # Compute advantages and targets
        # pass Vs without gradient information
        advantages, targets = self.compute_advantages(rewards_t, dones_t, Vs, T)

        # Update critic to minimize TD error (stays the same as A2C)
        self._update_critic(Vs, targets)

        # convert data to TensorDataset for minibatch sampling
        dataset = torch.utils.data.TensorDataset(
            states_t,
            torch.stack(old_log_probs_t),
            advantages,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            shuffle=True
        )

        for (states, old_log_probs, advs) in dataloader:
            # Compute new log probabilities
            new_log_probs = self.actor(states)
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
            log_prob_t = dist.log_prob(action_t)
            entropy_t = dist.entropy()

        # we need to save the log prob and entropy grads
        # for the update step
        return action_t.cpu().numpy(), log_prob_t, entropy_t

    def train(self):
        print(f"Starting training for A2C...")
        rewards_all = []

        for ep in range(self.num_episodes):
            env = self.env.get_env(ep)
            state, _ = env.reset()
            rewards, states, dones = [], [], []
            log_probs, entropies = [], []
            total: float = 0.0

            # Trajectory collection phase
            # collect rollout
            for t in range(self.max_steps):
                actions, log_probs_t, entropies_t = self.select_action(state)
                next_state, reward, done, trunc, _ = env.step(actions)

                log_probs.append(log_probs_t)
                entropies.append(entropies_t)
                rewards.append(reward)
                states.append(next_state)
                dones.append(np.array(done) | np.array(trunc))

                state = next_state

            # Update phase

            

            # convert to numpy array for better performance
            states = np.array(states)
            rewards = np.array(rewards)
            dones = np.array(dones)
            self.update(log_probs, entropies, rewards, states, dones)
            # TODO: Add entropy regularization call here
            # self.decay_exploration_param()
            rewards_all.append(total)
            self.print_progress(ep, total, rewards_all)

        return rewards_all

