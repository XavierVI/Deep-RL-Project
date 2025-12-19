import gymnasium as gym


class EnvironmentManager:
    """
    This class manages the environment setup.
    """
    def __init__(self, cfg):
        # Environment setup (may create vectorized envs)
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
            self.render_env = gym.make(
                cfg["env_name"],
                render_mode="human"
            )
            self.display = cfg.get("display", False)
            self.num_envs = 1
            self.render_int = cfg.get("display_episodes", 100)

        try:
            self.action_space_type = cfg["action_space_type"]
        except KeyError:
            raise ValueError(
                "action_space_type not specified in config. Please specify 'discrete' or 'continuous'.")

    def get_state_space_size(self):
        if hasattr(self.env, "single_observation_space"):
            return self.env.single_observation_space.shape[0]
        else:
            return self.env.observation_space.shape[0]

    def get_action_space_size(self):
        """
        Returns the size of an action space.

        For discrete action spaces, it returns the number of possible actions. For continuous action spaces, it returns the dimensionality of the action space.
        """
        # the dimension of the action space is found using
        # shape[0]
        if self.action_space_type == "continuous":
            if hasattr(self.env, "single_action_space"):
                return self.env.single_action_space.shape[0]
            else:
                return self.env.action_space.shape[0]
        # for discrete action spaces, it is found using n
        else:
            if hasattr(self.env, "single_action_space"):
                return self.env.single_action_space.n
            else:
                return self.env.action_space.n

    def get_env(self, episode):
        """Get the appropriate environment (render or non-render) for the episode."""
        return self.render_env if self.display and (episode + 1) % self.render_int == 0 else self.env

    def render(self, episode):
        """Render the environment if display is enabled."""
        if self.display and (episode + 1) % self.render_int == 0:
            self.env.render()
