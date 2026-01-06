import numpy as np
import torch
import torch.optim as optim
from abc import ABC, abstractmethod

from reinforcement_learning.networks import *
from reinforcement_learning.EnvironmentManager import EnvironmentManager
from reinforcement_learning.Exploration import ExplorationStrategy


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

    def _init_actor_critic(self, cfg):
        """Initialize actor-critic networks and optimizers."""
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
        self.critic_opt = optim.Adam(
            self.critic.parameters(), lr=cfg["learning_rate"])

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

    #--------------------------------------------------------------- #
    #               Advantage and Target Computation                 #
    #--------------------------------------------------------------- #

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
                gaes[t] = delta + self.gamma * self.lambda_ * \
                    (1 - dones_t[t]) * future_gae
                future_gae = gaes[t]

            # Target is V(s) + A(s,a)
            targets = Vs[:-1] + gaes
            return gaes, targets

    # --------------------------------------------------------------- #
    #                          Helper Methods                         #
    # --------------------------------------------------------------- #
    def save_model(self, filepath):
        """Save the actor and critic models to the specified filepath."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, filepath)