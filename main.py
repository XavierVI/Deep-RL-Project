
import os
import json
import argparse
import itertools
from copy import deepcopy
import numpy as np
import torch
import torch.multiprocessing as mp

from agents import *



def save_rewards_to_csv(rewards, lr, gamma, use_boltzmann):
    # save rewards data to a CSV file using numpy
    os.makedirs("results", exist_ok=True)
    filename = f"results/lr-{lr}_gamma-{gamma}_boltzmann-{use_boltzmann}.csv"
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

    # --------------------------------------------------------------- #
    #                     Setting Up the Trainer                    #
    # --------------------------------------------------------------- #
    agent_class_map = {
        "reinforce": ReinforceAgent,
        "sarsa": SarsaAgent,
        # "QLearningAgent": QLearningAgent
    }
    
    agent_class = agent_class_map[cfg["algorithm"]]
    trainer = agent_class(cfg)
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

    parser.add_argument(
        "--config_file", type=str, default="global_config.json",
        help="Path to the global configuration file."
    )

    args = parser.parse_args()

    if args.parallel:
        parallel_main(args.num_processes)
    else:
        main()
