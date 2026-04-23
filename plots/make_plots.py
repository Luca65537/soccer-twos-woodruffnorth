"""Generate training curve plots from Ray Tune progress.csv files."""
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

AGENTS = [
    ("Vanilla PPO", "vanilla_progress.csv", "#1f77b4"),
    ("Reward-Shaped PPO", "shaped_progress.csv", "#d62728"),
    ("Curriculum PPO", "curriculum_progress.csv", "#2ca02c"),
]


def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[["timesteps_total", "episode_reward_mean", "episode_len_mean"]].dropna()
    return df


def smooth(series, window=25):
    return series.rolling(window=window, min_periods=1).mean()


def plot_overlay():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, fname, color in AGENTS:
        df = load(os.path.join(HERE, fname))
        x = df["timesteps_total"] / 1e6
        ax1.plot(x, smooth(df["episode_reward_mean"]), label=label, color=color, linewidth=1.8)
        ax2.plot(x, smooth(df["episode_len_mean"]), label=label, color=color, linewidth=1.8)

    ax1.set_xlabel("Environment Steps (millions)")
    ax1.set_ylabel("Episode Reward Mean (smoothed)")
    ax1.set_title("Learning Curves — All Agents")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.axhline(0, color="gray", linewidth=0.5)

    ax2.set_xlabel("Environment Steps (millions)")
    ax2.set_ylabel("Episode Length (steps, smoothed)")
    ax2.set_title("Episode Length — All Agents")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    out = os.path.join(HERE, "comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close()


def plot_individual():
    for label, fname, color in AGENTS:
        df = load(os.path.join(HERE, fname))
        x = df["timesteps_total"] / 1e6

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(x, smooth(df["episode_reward_mean"]), color=color, linewidth=1.8, label="Reward (smoothed)")
        ax1.plot(x, df["episode_reward_mean"], color=color, alpha=0.25, linewidth=0.8, label="Reward (raw)")
        ax1.set_xlabel("Environment Steps (millions)")
        ax1.set_ylabel("Episode Reward Mean", color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color="gray", linewidth=0.5)

        ax2 = ax1.twinx()
        ax2.plot(x, smooth(df["episode_len_mean"]), color="#888888", linewidth=1.4, linestyle="--", label="Ep. Length (smoothed)")
        ax2.set_ylabel("Episode Length (steps)", color="#888888")
        ax2.tick_params(axis="y", labelcolor="#888888")

        ax1.set_title(f"{label} — Training Curve")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

        slug = fname.replace("_progress.csv", "")
        out = os.path.join(HERE, f"{slug}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close()


if __name__ == "__main__":
    plot_overlay()
    plot_individual()
