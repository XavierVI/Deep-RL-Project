import itertools
import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim

import gymnasium as gym
import numpy as np

from copy import deepcopy


"""
NOTES: 
- The actions for Lunar Lander are discrete: {0, 1, 2, 3}
  (source: https://gymnasium.farama.org/environments/box2d/lunar_lander/)

TODO:
- Implement Boltzmann action selection
"""

# --------------------------------------------------------------- #
#                     Policy Class Definition                     #
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


# --------------------------------------------------------------- #
#                     Agent Class Definition                     #
# --------------------------------------------------------------- #
class SarsaAgent:
    def __init__(self, cfg):
        # environment setup
        self.env = gym.make(cfg["env_name"], render_mode=None)
        self.render_env = gym.make(cfg["env_name"], render_mode="human")
        self.display = cfg.get("display", False)
        self.device = cfg.get("device", torch.device("cpu"))

        # state and action dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # hyperparameters
        self.gamma = cfg["gamma"]
        self.use_boltzmann = cfg.get("sarsa_use_boltzmann", False)

        # epsilon-greedy parameters
        self.epsilon = cfg["sarsa_initial_epsilon"]
        self.min_epsilon = cfg["sarsa_min_epsilon"]
        self.decay = cfg["sarsa_decay_rate"]
        
        # boltzmann parameters
        self.tau = cfg["sarsa_initial_tau"]
        self.min_tau = cfg["sarsa_min_tau"]
        self.decay_rate_tau = cfg["sarsa_decay_rate_tau"]

        self.render_int = cfg.get("display_episodes", 100)
        self.num_episodes = cfg["num_episodes"]
        self.max_steps = cfg.get("max_steps", 200)

        # policy network, optimizer, and loss function
        self.q_net = SarsaQNet(
            self.state_dim, self.action_dim,
            cfg["hidden_layers"], cfg["activation_function"]).to(self.device)

        self.opt = optim.Adam(self.q_net.parameters(), lr=cfg["learning_rate"])
        self.criterion = nn.MSELoss()

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

    def decay_epsilon(self):
        """
        Decay epsilon or tau.
        """
        if self.use_boltzmann:
            self.tau = max(
                self.min_tau, self.tau * np.exp(-self.decay_rate_tau))
        else:
            self.epsilon = max(
                self.min_epsilon, self.epsilon * np.exp(-self.decay))

    # Additional methods for training would go here
    def train(self):
        """
        SARSA training loop implementation.
        """
        print(f"Starting training with parameters gamma={self.gamma}, use_boltzmann={self.use_boltzmann}...")
        rewards_all = []

        for m in range(self.num_episodes):
            # total rewards for this episode
            total = 0

            env = self.render_env \
                if self.display and (m + 1) % self.render_int == 0 \
                else self.env
            
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
            # Convert experiences to batched tensors for parallel processing
            # First convert to numpy arrays, then to tensors (faster)
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

            # Compute all Q-values in parallel (batched forward pass)
            q_values = self.q_net(states)
            # print("----- Q VALUES -----")
            # print(q_values.size())
            # Gather the predicted Q-values for the taken actions (actions.unsqueeze(1))
            pred_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # print("----- PREDICTED Q VALUES -----")
            # print(pred_q_values.size())

            # Compute target Q-values in parallel
            with torch.no_grad():
                next_q_values = self.q_net(next_states)
                next_q_for_actions = next_q_values.gather(
                    1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + self.gamma * next_q_for_actions * (1 - dones)

            # Single backward pass for entire episode
            loss = self.criterion(pred_q_values, targets)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            rewards_all.append(total)
            if self.use_boltzmann and (m + 1) % self.render_int == 0:
                print(
                    f"Episode {m+1}/{self.num_episodes} | Reward {total:.1f} | Avg {np.mean(rewards_all[-20:]):.1f} | τ={self.tau:.3f}", flush=True)
            elif (m + 1) % self.render_int == 0:
                print(
                    f"Episode {m+1}/{self.num_episodes} | Reward {total:.1f} | Avg {np.mean(rewards_all[-20:]):.1f} | ε={self.epsilon:.3f}", flush=True)

            self.decay_epsilon()

        return rewards_all

def save_rewards_to_csv(rewards, lr, gamma, use_boltzmann):
    # save rewards data to a CSV file using numpy
    os.makedirs("results", exist_ok=True)
    filename = f"results/sarsa_lr-{lr}_gamma-{gamma}_boltzmann-{use_boltzmann}.csv"
    exists = os.path.exists(filename)
    
    # Create array with episodes and rewards
    episodes = np.arange(1, len(rewards) + 1)
    data = np.column_stack((episodes, rewards))
    
    with open(filename, 'a') as f:
        if not exists:
            # write header once
            np.savetxt(f, data, delimiter=',', header='episode,reward',
                       comments='', fmt='%d,%.6f')
        else:
            # just append rows (no header)
            np.savetxt(f, data, delimiter=',', comments='', fmt='%d,%.6f')

    print(f"Saved rewards to {filename}")


def run_training(lr, gamma, use_boltzmann, cfg):
    trials = 5
    print(
        f"Training with lr={lr}, gamma={gamma}, use_boltzmann={use_boltzmann}")

    cfg["learning_rate"] = lr
    cfg["gamma"] = gamma
    cfg["sarsa_use_boltzmann"] = use_boltzmann
    base_seed = cfg["seed"]

    for trial in range(trials):
        np.random.seed(base_seed + trial)
        torch.manual_seed(base_seed + trial)
        trainer = SarsaAgent(cfg)
        rewards = trainer.train()
        save_rewards_to_csv(rewards, lr, gamma, use_boltzmann)
    

def parallel_main(processes=os.cpu_count()):
    """
    This will run the trainer in parallel with different hyperparameters.

    @param processes: Number of parallel processes to use in the processing pool.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "global_config.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # add device to cfg
    cfg["device"] = device

    args = [
        (lr, gamma, use_boltzmann, deepcopy(cfg))
        for lr, gamma, use_boltzmann in itertools.product(
            cfg["learning_rates"],
            cfg["gammas"],
            cfg["boltzmann_options"]
        )
    ]

    # specify start method: each child starts fresh and only receives
    # the given arguments.
    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes) as pool:
        pool.starmap(run_training, args)


def main():
    """
    This will run the trainer with the hyperparameters from the JSON file.
    """

    # --------------------------------------------------------------- #
    #                     Loading JSON Config File                    #
    # --------------------------------------------------------------- #
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "global_config.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # add device to cfg
    cfg["device"] = device
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    
    trainer = SarsaAgent(cfg)
    rewards = trainer.train()
    
    save_rewards_to_csv(rewards, cfg["learning_rate"], cfg["gamma"], cfg.get("use_boltzmann", False))


if __name__ == "__main__":
    """
    usage: python sarsa.py [--parallel] [--num_processes=<int>]
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--parallel", action="store_true",
        help="Run training in parallel with different hyperparameters."
    )

    
    parser.add_argument(
        "--num_processes", type=int, default=2,
        help="Number of processes to use for parallel training."
    )

    args = parser.parse_args()

    if args.parallel:
        parallel_main(args.num_processes)
    else:
        main()
