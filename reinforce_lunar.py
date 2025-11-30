import argparse
import itertools
import gymnasium as gym
import numpy as np
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from copy import deepcopy


# --- Policy network definition ---
class PolicyNet(nn.Module):
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


# --- REINFORCE agent ---
class ReinforceAgent:
    def __init__(self, cfg):
        self.env = gym.make(cfg["env_name"], render_mode=None)
        self.render_env = gym.make(cfg["env_name"], render_mode="human")
        self.display = cfg["display"]
        self.device = cfg["device"]
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        self.gamma = cfg["gamma"]
        self.use_boltzmann = cfg["reinforce_use_boltzmann"]

        # epsilon-greedy parameters
        self.epsilon = cfg["reinforce_initial_epsilon"]
        self.min_epsilon = cfg["reinforce_min_epsilon"]
        self.decay = cfg["reinforce_decay_rate"]
        
        # boltzmann parameters
        self.tau = cfg["reinforce_initial_tau"]
        self.min_tau = cfg["reinforce_min_tau"]
        self.decay_rate_tau = cfg["reinforce_decay_rate_tau"]

        self.render_int = cfg["display_episodes"]
        self.use_baseline = cfg["reinforce_use_baseline"]

        self.episodes = cfg["num_episodes"]
        self.max_steps = cfg["max_steps"]

        self.policy = PolicyNet(
            self.state_dim,
            self.action_dim,
            cfg["hidden_layers"],
            cfg["activation_function"]
        ).to(self.device)
        self.opt = optim.Adam(
            self.policy.parameters(),
            lr=cfg["learning_rate"]
        )

        if self.use_baseline:
            self.baseline = nn.Linear(self.state_dim, 1).to(self.device)
            self.baseline_opt = optim.Adam(self.baseline.parameters(), lr=cfg["learning_rate"])

    def select_action(self, state):
        # fetch the action probabilities from the policy network
        # (probability for each action)
        logits_t, probs_t = self.policy(
            torch.FloatTensor(state).unsqueeze(0).to(self.device)
        )
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

    def update(self, log_probs, rewards, states):
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


    def train(self):
        print(f"Starting training with parameters gamma={self.gamma}, use_boltzmann={self.use_boltzmann}...")
        rewards_all = []

        for ep in range(self.episodes):
            env = self.render_env \
                if (ep + 1) % self.render_int == 0 and self.display \
                else self.env
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

            self.update(log_probs, rewards, states)
            self.decay_epsilon()
            rewards_all.append(total)
            
            if self.use_boltzmann and (ep + 1) % self.render_int == 0:
                print(
                    f"Episode {ep+1}/{self.episodes} | Reward {total:.1f} | Avg {np.mean(rewards_all[-20:]):.1f} | τ={self.tau:.3f}", flush=True)
            elif (ep + 1) % self.render_int == 0:
                print(
                    f"Episode {ep+1}/{self.episodes} | Reward {total:.1f} | Avg {np.mean(rewards_all[-20:]):.1f} | ε={self.epsilon:.3f}", flush=True)

        return rewards_all


def save_rewards_to_csv(rewards, lr, gamma, use_boltzmann):
    # save rewards data to a CSV file using numpy
    os.makedirs("results", exist_ok=True)
    filename = f"results/reinforce_lr-{lr}_gamma-{gamma}_boltzmann-{use_boltzmann}.csv"
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
    cfg["reinforce_use_boltzmann"] = use_boltzmann
    base_seed = cfg["seed"]

    for trial in range(trials):
        np.random.seed(base_seed + trial)
        torch.manual_seed(base_seed + trial)
        trainer = ReinforceAgent(cfg)
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

    trainer = ReinforceAgent(cfg)
    rewards = trainer.train()

    save_rewards_to_csv(
        rewards, cfg["learning_rate"], cfg["gamma"], cfg.get("use_boltzmann", False))


if __name__ == "__main__":
    """
    usage: python lunear_reinforce.py [--parallel] [--num_processes=<int>]
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
