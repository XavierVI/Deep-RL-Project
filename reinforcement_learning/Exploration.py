import numpy as np

class ExplorationStrategy:
    """
    Class to handle exploration parameter decay.
    
    The exploration strategy is specified using:
    "exp_strat": "epsilon" for epsilon-greedy,
    "exp_strat": "boltzmann" for Boltzmann exploration,
    "exp_strat": "entropy_reg" for entropy regularization.
    """
    def __init__(self, cfg):
        try:
            self.exp_strat = cfg["exp_strat"]
        except KeyError:
            raise ValueError("exp_strat not specified in config. Please specify 'epsilon', 'boltzmann', or 'entropy_reg'.")

        if self.exp_strat == "epsilon":
            self._setup_epsilon_params(cfg)
        elif self.exp_strat == "boltzmann":
            self._setup_boltzmann_params(cfg)
        elif self.exp_strat == "entropy_reg":
            self._setup_entropy_params(cfg)
        else:
            raise ValueError("Invalid exp_strat specified. Choose 'epsilon', 'boltzmann', or 'entropy_reg'.")


    def _setup_epsilon_params(self, cfg):
        try:
            self.epsilon = cfg["initial_epsilon"]
            self.min_epsilon = cfg["min_epsilon"]
            self.decay = cfg["decay_rate"]
        except KeyError:
            raise ValueError(
                "Missing epsilon-greedy parameters in config.")

    def _setup_boltzmann_params(self, cfg):
        try:
            self.tau = cfg["initial_tau"]
            self.min_tau = cfg["min_tau"]
            self.decay_rate_tau = cfg["decay_rate_tau"]
        except KeyError:
            raise ValueError(
                "Missing Boltzmann parameters in config.")

    def _setup_entropy_params(self, cfg):
        try:
            self.beta = cfg["beta"]
            self.min_beta = cfg["min_beta"]
            self.entropy_decay = cfg["entropy_decay"]
        except KeyError:
            raise ValueError(
                "Missing entropy regularization parameters in config.")

    def decay_exploration_param(self):
        """Decay epsilon or tau."""
        if self.exp_strat == "boltzmann":
            self.tau = max(self.min_tau, self.tau *
                            np.exp(-self.decay_rate_tau))
        elif self.exp_strat == "epsilon":
            self.epsilon = max(self.min_epsilon, self.epsilon * np.exp(-self.decay))

        elif self.exp_strat == "entropy_reg":
            self.beta = max(self.min_beta, self.beta * np.exp(-self.entropy_decay))

    def get_exploration_param(self):
        """Get the current exploration parameter."""
        if self.exp_strat == "boltzmann":
            return self.tau
        elif self.exp_strat == "epsilon":
            return self.epsilon
        elif self.exp_strat == "entropy_reg":
            return self.beta

    def get_strategy(self) -> str:
        """Get the exploration strategy type."""
        return self.exp_strat