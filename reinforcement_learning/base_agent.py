import numpy as np
import torch
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
