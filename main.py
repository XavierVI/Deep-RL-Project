
import os
import json
import argparse
import itertools
from copy import deepcopy
import numpy as np
import torch
import torch.multiprocessing as mp
import csv

from agents import *


def init_profiler(wait: int = 1, warmup: int = 1, active: int = 2):
    """Create a PyTorch profiler to capture CPU and CUDA function runtimes.
    
    Uses a schedule to control when profiling starts and stops. Call prof.step()
    after each training iteration to advance through the schedule.
    
    Args:
        wait: Number of initial steps to skip before warmup
        warmup: Number of warmup steps to skip profiling data
        active: Number of steps to actively profile and record
    
    Returns:
        torch.profiler.profile: A configured profiler context manager.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        # Include CUDA kernel timing when CUDA is present
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active)
    
    return torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    )


def export_profiler_artifacts(prof, out_dir: str, run_name: str = "run"):
    os.makedirs(out_dir, exist_ok=True)

    averages = prof.key_averages()
    output_path = os.path.join(out_dir, f"{run_name}_summary.csv")

    with open(output_path, "w") as f:
        # Header
        f.write(
            "name,count,cpu_time,device_time\n")

        for entry in averages:
            # Use stable properties; truncate name to max 10 chars
            name = str(getattr(entry, "key", getattr(entry, "name", "")))[:10]
            # units are reported in microseconds; convert to seconds for CSV
            f.write(f'"{name}",' 
                f"{entry.count},"
                f"{entry.cpu_time / 1_000_000},"
                f"{entry.device_time / 1_000_000}\n")

    print(f"Successfully dumped profiler data to {output_path}")


def save_rewards_to_csv(rewards, lr, gamma, use_boltzmann, file_name=None):
    # save rewards data to a CSV file using numpy
    os.makedirs("results", exist_ok=True)
    if file_name is not None:
        filename = file_name
    else:
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


def main(config_file_name, profile=False):
    """
    This will run the trainer with the hyperparameters from the JSON file.
    """

    # --------------------------------------------------------------- #
    #                     Loading JSON Config File                    #
    # --------------------------------------------------------------- #
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", config_file_name)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 50)
    print("*")
    print(f"* Using config file: {config_path}")
    print(f"* Using device: {device}")
    print("*")
    print("=" * 50)
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
        "a2c": A2CAgent,
        # "QLearningAgent": QLearningAgent
    }

    if profile:
        run_name = config_file_name.replace('.json', '')
        log_dir = f"./profiler/{run_name}"
        os.makedirs(log_dir, exist_ok=True)
        with init_profiler() as prof:
            agent_class = agent_class_map[cfg["algorithm"]]
            trainer = agent_class(cfg)
            rewards = trainer.train(profiler=prof)
        export_profiler_artifacts(prof, out_dir=log_dir, run_name=run_name)
    else:
        agent_class = agent_class_map[cfg["algorithm"]]
        trainer = agent_class(cfg)
        rewards = trainer.train()
    
    save_rewards_to_csv(
        rewards,
        cfg["learning_rate"],
        cfg["gamma"],
        cfg.get("use_boltzmann",
        False),
        file_name=f"results/{config_file_name.replace('.json', '')}_rewards.csv"
    )


if __name__ == "__main__":
    """
    usage: python main.py [--parallel] [--num_processes=<int>]
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
        "--config", type=str, default="global_config.json",
        help="Name of the configuration file in the config directory."
    )

    # disabled because profiler doesn't work very well
    # parser.add_argument(
    #     "--profile", action="store_true",
    #     help="Enable PyTorch profiler during training."
    # )

    args = parser.parse_args()

    if args.parallel:
        parallel_main(args.num_processes)
    else:
        main(args.config, args.profile)
