import argparse
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

"""
This file is used to create a plot of the results obtained from training the agents.


"""

# modify default font size
plt.rcParams.update({'font.size': 26})

fig_size = (20, 18)

linewidth = 2.5

color_map = {
    "0.01": "red",
    "0.001": "blue",
    "0.0001": "green"
}

row_map = {
    "0.9": 0,
    "0.95": 1,
    "0.99": 2
}

def extract_hyperparameters(filename):
    """
    This function extracts hyperparameters from the filename.
    The expected filename format is:
    alg_lr-<learning_rate>_gamma-<discount_factor>_boltzmann-<True|False>.csv
    """
    parts = filename.split("_")

    learning_rate = parts[1].split("-")[1]
    discount_factor = parts[2].split("-")[1]
    boltzmann = parts[3].split("-")[1].replace(".csv", "")

    return (learning_rate, discount_factor, boltzmann)


def read_csv_files(results_path, alg):
    """
    This function simply reads all the CSV files in results_path,
    and returns the data as a dictionary of numpy arrays.
    """

    data = {}

    for filename in os.listdir(results_path):
        if filename.endswith(".csv") and alg in filename:
            # read the csv file using numpy, skip header row
            file_path = os.path.join(results_path, filename)
            file_data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            lr, gamma, boltz = extract_hyperparameters(filename)
            data.setdefault(boltz, {}).setdefault(gamma, {}).setdefault(lr, file_data)

    return data


def average_and_std_by_episode(file_data):
    """
    Group rewards by episode index and return episode numbers, averaged rewards,
    and standard deviation per episode.

    file_data is expected to be an (N,2) array with columns [episode, reward].
    Multiple rows can share the same episode index; this function computes mean and std.
    """
    if file_data is None or file_data.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Ensure 2D
    if file_data.ndim == 1:
        file_data = file_data.reshape(1, -1)

    episodes = file_data[:, 0].astype(int)
    rewards = file_data[:, 1].astype(float)

    # Fetch unique episodes, and the indices for each episode
    unique_eps, inverse = np.unique(episodes, return_inverse=True)
    sum_rewards = np.zeros_like(unique_eps, dtype=float)
    sumsq_rewards = np.zeros_like(unique_eps, dtype=float)
    counts = np.zeros_like(unique_eps, dtype=int)

    # accumulate sums and squared sums for mean/std
    np.add.at(sum_rewards, inverse, rewards)
    np.add.at(sumsq_rewards, inverse, rewards ** 2)
    np.add.at(counts, inverse, 1)

    means = sum_rewards / counts
    # variance via E[x^2] - E[x]^2
    variances = (sumsq_rewards / counts) - (means ** 2)
    # numerical safety
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances)

    return unique_eps, means, stds


def moving_average(x, window=20):
    """
    Compute a moving average with given window.

    Uses 'same' convolution so edges are smoothed with smaller effective window.
    """
    w = np.ones(window, dtype=float) / window
    return np.convolve(x, w, mode='same')


def create_plot(data, alg, used_boltzmann):
    fig, axes = plt.subplots(3, 1, figsize=fig_size)

    for gamma in data:
        for lr in data[gamma]:
            file_data = data[gamma][lr]
            unique_eps = np.unique(file_data[:, 0])
            trial_length = unique_eps.shape[0]

            # only use the first trial
            eps = file_data[:trial_length, 0].astype(int)
            rewards = file_data[:trial_length, 1].astype(float)

            axes[row_map[gamma]].plot(
                eps,
                rewards,
                color=color_map[lr],
                linewidth=linewidth
            )

    for ax in axes:
        # ax.set_yticks(np.arange(-1500, 10, 500))
        ax.grid(True)

    axes[0].set_ylabel("Reward ($\gamma$=0.9)")
    axes[1].set_ylabel("Reward ($\gamma$=0.95)")
    axes[2].set_ylabel("Reward ($\gamma$=0.99)")

    axes[-1].set_xlabel("Episodes")

    # Create a custom legend for the learning rate colors
    # sort learning rates numerically (largest to smallest) for consistent legend order
    lr_keys = sorted(color_map.keys(), key=lambda x: float(x), reverse=True)
    legend_handles = [
        Line2D([0], [0], color=color_map[k], lw=linewidth) for k in lr_keys]
    legend_labels = [f"lr={k}" for k in lr_keys]
    # place a single legend for the whole figure
    fig.legend(legend_handles, legend_labels, loc='upper right', fontsize=18)
    fig.tight_layout()
    fig.savefig(f'./results/{alg}_rewards_plot_{used_boltzmann}.png', dpi=300)
    print(f"Saved plot to ./results/{alg}_rewards_plot_{used_boltzmann}.png")
    plt.close(fig)


def create_ma_plot(data, alg, used_boltzmann):
    fig, axes = plt.subplots(3, 1, figsize=fig_size)

    for gamma in data:
        for lr in data[gamma]:
            file_data = data[gamma][lr]
            eps, means, stds = average_and_std_by_episode(file_data)

            # smooth with 20-episode moving average
            ma_means = moving_average(means, window=20)
            ma_stds = moving_average(stds, window=20)

            axes[row_map[gamma]].plot(
                eps,
                ma_means,
                color=color_map[lr],
                linewidth=linewidth
            )
            # add shaded area for +/- 1 std (smoothed)
            axes[row_map[gamma]].fill_between(
                eps,
                ma_means - ma_stds,
                ma_means + ma_stds,
                color=color_map[lr],
                alpha=0.2
            )

    for ax in axes:
        # ax.set_yticks(np.arange(-550, 10, 500))
        ax.grid(True)

    axes[0].set_ylabel("Reward ($\gamma$=0.9)")
    axes[1].set_ylabel("Reward ($\gamma$=0.95)")
    axes[2].set_ylabel("Reward ($\gamma$=0.99)")

    axes[-1].set_xlabel("Episodes")

    # Create a custom legend for the learning rate colors
    # sort learning rates numerically (largest to smallest) for consistent legend order
    lr_keys = sorted(color_map.keys(), key=lambda x: float(x), reverse=True)
    legend_handles = [Line2D([0], [0], color=color_map[k], lw=linewidth) for k in lr_keys]
    legend_labels = [f"lr={k}" for k in lr_keys]
    # place a single legend for the whole figure
    fig.legend(legend_handles, legend_labels, loc='upper right', fontsize=18)
    fig.tight_layout()
    fig.savefig(f'./results/{alg}_ma_rewards_plot_{used_boltzmann}.png', dpi=300)
    plt.close(fig)


def plot_rewards(data, alg):
    """
    Plot the total rewards over episodes.
    """
    print(f"Generating plots for {alg}...")
    for boltz in data:
        create_plot(
            data[boltz],
            alg,
            "boltzmann" if boltz == "True" else "epsilon_greedy"
        )


def create_side_by_side_plot(data_sarsa, data_reinforce, boltz_label, moving_avg=False):
    """
    Create a side-by-side comparison plot for SARSA (left column) and REINFORCE (right column).

    data_sarsa and data_reinforce should be dictionaries mapping gamma -> lr -> file_data
    (the same format returned by read_csv_files but for a single boltz value).
    boltz_label is a string used for naming the output (e.g., 'True' or 'False').
    """
    fig, axes = plt.subplots(3, 2, figsize=fig_size, sharex=True, sharey='row')

    # columns: 0 -> SARSA, 1 -> REINFORCE
    alg_map = {0: ('REINFORCE', data_reinforce), 1: ('SARSA', data_sarsa)}

    # iterate rows by gamma in the same order as row_map
    ordered_gammas = sorted(row_map.keys(), key=lambda x: float(x))

    best_sarsa = {
        "reward": -float('inf'),
        "std_dev": None,
        "boltz": None,
        "lr": None,
        "gamma": None
    }

    best_reinforce = {
        "reward": -float('inf'),
        "std_dev": None,
        "boltz": None,
        "lr": None,
        "gamma": None
    }

    for row_idx, gamma in enumerate(ordered_gammas):
        for col_idx in (0, 1):
            alg_name, alg_data = alg_map[col_idx]
            # if this algorithm has no data for this boltz/gamma, skip
            if gamma not in alg_data:
                continue

            for lr in alg_data[gamma]:
                file_data = alg_data[gamma][lr]
                if moving_avg:
                    eps, means, stds = average_and_std_by_episode(file_data)
                    if eps.size == 0:
                        continue

                    window = 20
                    ma_means = moving_average(means, window=window)
                    ma_stds = moving_average(stds, window=window)
                    # if len(means) >= window:
                        # eps_ma = eps[window - 1:]
                    # else:
                        # eps_ma = np.array([eps[-1]])
                    
                    
                    if alg_name == "SARSA" and ma_means[-1] > best_sarsa["reward"]:
                        best_sarsa["reward"] = ma_means[-1]
                        best_sarsa["std_dev"] = ma_stds[-1]
                        best_sarsa["boltz"] = boltz_label
                        best_sarsa["lr"] = lr
                        best_sarsa["gamma"] = gamma

                    if alg_name == "REINFORCE" and ma_means[-1] > best_reinforce["reward"]:
                        best_reinforce["reward"] = ma_means[-1]
                        best_reinforce["std_dev"] = ma_stds[-1]
                        best_reinforce["boltz"] = boltz_label
                        best_reinforce["lr"] = lr
                        best_reinforce["gamma"] = gamma


                    ax = axes[row_idx, col_idx]
                    ax.plot(eps, ma_means, color=color_map[lr], linewidth=linewidth)
                    ax.fill_between(eps, ma_means - ma_stds, ma_means + ma_stds,
                                    color=color_map[lr], alpha=0.2)

                else:                    
                    eps, rewards, std_dev = average_and_std_by_episode(file_data)

                    ax = axes[row_idx, col_idx]
                    ax.plot(
                        eps,
                        rewards,
                        color=color_map[lr],
                        linewidth=linewidth
                    )
                    ax.fill_between(eps, rewards - std_dev, rewards + std_dev,
                                    color=color_map[lr], alpha=0.2)

            ax = axes[row_idx, col_idx]
            # ax.set_yticks(np.arange(-1500, 10, 500))
            ax.grid(True)
    
    
    axes[0, 0].set_title("REINFORCE")
    axes[0, 1].set_title("SARSA")

    # set common x-label on bottom row
    axes[-1, 0].set_xlabel("Episodes")
    axes[-1, 1].set_xlabel("Episodes")

    # set common y-label on left column
    axes[0, 0].set_ylabel("Reward ($\gamma$=0.9)")
    axes[1, 0].set_ylabel("Reward ($\gamma$=0.95)")
    axes[2, 0].set_ylabel("Reward ($\gamma$=0.99)")


    # build legend handles from color_map
    lr_keys = sorted(color_map.keys(), key=lambda x: float(x), reverse=True)
    legend_handles = [Line2D([0], [0], color=color_map[k], lw=linewidth) for k in lr_keys]
    legend_labels = [f"lr={k}" for k in lr_keys]
    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=len(lr_keys))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs('results', exist_ok=True)
    if moving_avg:
        fname = f"results/compare_sarsa_reinforce_ma_{'boltz' if boltz_label=='True' else 'eps'}.png"
    else:
        fname = f"results/compare_sarsa_reinforce_{'boltz' if boltz_label=='True' else 'eps'}.png"
    fig.savefig(fname, dpi=300)
    print(f"Saved comparison plot to {fname}")

    if moving_avg:
        print("Converged average rewards with moving average:")
        print(f"Best SARSA (boltz={best_sarsa['boltz']}, lr={best_sarsa['lr']}, gamma={best_sarsa['gamma']}): Avg. reward: {best_sarsa['reward']:.2f} +/- {best_sarsa['std_dev']:.2f}")
        print(f"Best REINFORCE (boltz={best_reinforce['boltz']}, lr={best_reinforce['lr']}, gamma={best_reinforce['gamma']}): Avg. reward: {best_reinforce['reward']:.2f} +/- {best_reinforce['std_dev']:.2f}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="./results", help="Path to the results directory")
    parser.add_argument(
        "--alg", type=str, default="side-by-side",
        choices=["both", "sarsa", "reinforce", "side-by-side"],
        help="The algorithm to generate plots for. Options: 'both', 'sarsa', 'reinforce'" )
    args = parser.parse_args()

    # make results directory if it doesn't exist
    os.makedirs(args.results, exist_ok=True)

    if args.alg == "both":
        # Generate plots for both algorithms
        data_sarsa = read_csv_files(args.results, "sarsa")
        plot_rewards(data_sarsa, "sarsa")
        data_reinforce = read_csv_files(args.results, "reinforce")
        plot_rewards(data_reinforce, "reinforce")
    
    elif args.alg == "sarsa":
        # Generate plots for SARSA algorithm
        data = read_csv_files(args.results, "sarsa")
        plot_rewards(data, "sarsa")
    
    elif args.alg == "reinforce":
        # Generate plots for REINFORCE algorithm
        data = read_csv_files(args.results, "reinforce")
        plot_rewards(data, "reinforce")

    elif args.alg == "side-by-side":
        # Generate side-by-side comparison plots
        data_sarsa = read_csv_files(args.results, "sarsa")         # dict: boltz -> gamma -> lr -> data
        data_reinforce = read_csv_files(args.results, "reinforce")

        # For each boltz setting present in both datasets:
        for boltz in data_sarsa:
            if boltz in data_reinforce:
                create_side_by_side_plot(data_sarsa[boltz], data_reinforce[boltz], boltz)
                create_side_by_side_plot(data_sarsa[boltz], data_reinforce[boltz], boltz, moving_avg=True)
