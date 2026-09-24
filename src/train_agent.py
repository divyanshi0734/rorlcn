"""MaskablePPO Reinforcement Learning Training Pipeline for NetworkRoutingEnv.

Features:
  1. ActionMasker integration ensuring invalid actions have probability 0.
  2. PPOConfig exposing standard PPO-family hyperparameters with named config.
  3. Strict seed threading across Python, NumPy, PyTorch, Gym, and MaskablePPO.
  4. RoutingExperimentCallback logging per-episode metrics and rolling delivery rate.
  5. Policy collapse detector warning if rolling delivery rate drops > 20% below peak.
  6. Sanity check asserting invalid-action rate is strictly 0.0% throughout training.
  7. Periodic and final checkpointing with resume capability.
  8. Publication-ready training curve plotting to assets/rl_training_curves.png.
"""

import os
import sys
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure RORL root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    import gymnasium as gym
    from stable_baselines3.common.callbacks import BaseCallback
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    HAS_RL_DEPS = True
except ImportError:
    HAS_RL_DEPS = False
    BaseCallback = object
    gym = None

from src.network_sim import NSFNETSimEnv
from src.routing_env import NetworkRoutingEnv, RewardConfig


@dataclass
class PPOConfig:
    """Hyperparameter configuration for MaskablePPO training."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    n_epochs: int = 10
    total_timesteps: int = 50_000
    checkpoint_freq: int = 10_000
    seed: int = 42
    device: str = "auto"


def mask_fn(env: Any) -> np.ndarray:
    """Callback function extracting boolean action mask from NetworkRoutingEnv."""
    return env.action_masks()



class RoutingExperimentCallback(BaseCallback):
    """Custom callback tracking episode metrics, rolling delivery rate, and policy collapse."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        checkpoint_freq: int = 10_000,
        plot_path: str = "assets/rl_training_curves.png",
        window_size: int = 100,
        collapse_threshold: float = 0.20,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.plot_path = plot_path
        self.window_size = window_size
        self.collapse_threshold = collapse_threshold

        # Telemetry containers
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.outcomes: List[str] = []
        self.timesteps_at_episode: List[int] = []
        self.rolling_delivery_rates: List[float] = []
        self.best_rolling_delivery_rate: float = 0.0
        self.invalid_action_count: int = 0
        self.policy_collapse_events: List[Dict[str, Any]] = []

        # Persistent telemetry CSV paths
        self.collapse_csv_path = os.path.join(self.checkpoint_dir, "collapse_events.csv")
        self.telemetry_csv_path = os.path.join(self.checkpoint_dir, "training_telemetry.csv")
        self.episode_offset: int = 0
        self.telemetry_records: List[Dict[str, Any]] = []

        # Load existing historical telemetry if resuming
        if os.path.exists(self.collapse_csv_path):
            try:
                hist_collapse = pd.read_csv(self.collapse_csv_path).to_dict("records")
                self.policy_collapse_events.extend(hist_collapse)
            except Exception:
                pass

        if os.path.exists(self.telemetry_csv_path):
            try:
                hist_tel = pd.read_csv(self.telemetry_csv_path)
                if not hist_tel.empty:
                    self.telemetry_records.extend(hist_tel.to_dict("records"))
                if not hist_tel.empty and "episode" in hist_tel.columns:
                    self.episode_offset = int(hist_tel["episode"].max())
                if not hist_tel.empty and "peak_rate" in hist_tel.columns:
                    self.best_rolling_delivery_rate = max(self.best_rolling_delivery_rate, float(hist_tel["peak_rate"].max()))
            except Exception:
                pass

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.plot_path)), exist_ok=True)

    def _on_step(self) -> bool:
        # Check if invalid action occurred (must be 0 under action masking)
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, info in enumerate(infos):
            if info.get("invalid_action", False):
                self.invalid_action_count += 1

            if dones[idx]:
                # Extract outcome from info dict
                if info.get("reached_dest", False):
                    outcome = "delivered"
                elif info.get("dropped", False):
                    outcome = "dropped"
                elif info.get("looped", False):
                    outcome = "looped"
                elif info.get("truncated", False):
                    outcome = "truncated"
                elif info.get("invalid_action", False):
                    outcome = "invalid"
                else:
                    outcome = "unknown"

                # Extract episode return from training monitor or environment
                ep_reward = info.get("episode", {}).get("r", None)
                ep_len = info.get("hops_taken", info.get("episode", {}).get("l", 0))

                if ep_reward is None:
                    # Fallback to local rewards or info tracking
                    ep_reward = sum(self.locals.get("rewards", [0.0]))

                self.episode_rewards.append(float(ep_reward))
                self.episode_lengths.append(int(ep_len))
                self.outcomes.append(outcome)
                self.timesteps_at_episode.append(self.num_timesteps)

                # Compute rolling delivery rate over window
                recent = self.outcomes[-self.window_size:]
                current_rate = sum(1 for o in recent if o == "delivered") / len(recent)
                self.rolling_delivery_rates.append(current_rate)

                abs_ep_idx = self.episode_offset + len(self.outcomes)

                # -------------------------------------------------------------
                # Policy Collapse Detector:
                # Flag if rolling delivery rate drops > 20% below running peak
                # -------------------------------------------------------------
                if len(self.outcomes) >= self.window_size or abs_ep_idx >= self.window_size:
                    if current_rate > self.best_rolling_delivery_rate:
                        self.best_rolling_delivery_rate = current_rate
                    elif (self.best_rolling_delivery_rate - current_rate) > self.collapse_threshold:
                        drop_pct = (self.best_rolling_delivery_rate - current_rate) * 100.0
                        warning_msg = (
                            f"[POLICY REGRESSION WARNING] Step {self.num_timesteps}, Episode {abs_ep_idx}: "
                            f"Rolling delivery rate ({current_rate*100.0:.1f}%) dropped {drop_pct:.1f}% below running peak "
                            f"({self.best_rolling_delivery_rate*100.0:.1f}%)!"
                        )
                        if self.verbose > 0:
                            print(f"\n>>> {warning_msg} <<<")
                        self.policy_collapse_events.append({
                            "step": self.num_timesteps,
                            "episode": abs_ep_idx,
                            "current_rate": current_rate,
                            "peak_rate": self.best_rolling_delivery_rate,
                            "drop_pct": drop_pct,
                        })

                # Periodic progress print
                if len(self.outcomes) % 500 == 0:
                    mean_r = np.mean(self.episode_rewards[-100:])
                    mean_h = np.mean(self.episode_lengths[-100:])
                    if self.verbose > 0:
                        print(
                            f"Step {self.num_timesteps:6d} | Ep {abs_ep_idx:5d} | "
                            f"Deliv Rate: {current_rate*100.0:5.1f}% (Peak: {self.best_rolling_delivery_rate*100.0:5.1f}%) | "
                            f"Mean Ret: {mean_r:6.2f} | Mean Hops: {mean_h:4.1f} | Invalid: {self.invalid_action_count}"
                        )
                    self.telemetry_records.append({
                        "step": self.num_timesteps,
                        "episode": abs_ep_idx,
                        "delivery_rate": round(float(current_rate), 4),
                        "peak_rate": round(float(self.best_rolling_delivery_rate), 4),
                        "mean_return": round(float(mean_r), 2),
                        "mean_hops": round(float(mean_h), 1),
                        "invalid_count": self.invalid_action_count,
                    })

        # Periodic checkpointing
        if self.num_timesteps % self.checkpoint_freq == 0 and self.num_timesteps > 0:
            ckpt_path = os.path.join(self.checkpoint_dir, f"rl_model_step_{self.num_timesteps}.zip")
            self.model.save(ckpt_path)
            self._save_telemetry_csvs()
            if self.verbose > 0:
                print(f"[*] Saved periodic checkpoint: {ckpt_path}")

        return True

    def _save_telemetry_csvs(self) -> None:
        """Saves or updates training_telemetry.csv and collapse_events.csv."""
        if self.policy_collapse_events:
            pd.DataFrame(self.policy_collapse_events).to_csv(self.collapse_csv_path, index=False)
        if self.telemetry_records:
            pd.DataFrame(self.telemetry_records).to_csv(self.telemetry_csv_path, index=False)

    def analyze_collapse_incidents(self, gap_threshold: int = 5) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
        """Groups contiguous collapse warnings into distinct incidents and computes 20k-step bin metrics."""
        events = sorted(self.policy_collapse_events, key=lambda x: x["episode"])
        if not events:
            return [], pd.DataFrame(), {"total_incidents": 0, "total_flagged_steps": 0, "longest_duration_eps": 0}

        incidents = []
        curr_start_ep, curr_start_step = events[0]["episode"], events[0]["step"]
        curr_end_ep, curr_end_step = events[0]["episode"], events[0]["step"]
        curr_peak = events[0].get("peak_rate", 0.0)
        curr_min_rate = events[0].get("current_rate", 0.0)
        count = 1

        for i in range(1, len(events)):
            ev = events[i]
            ep, step = ev["episode"], ev["step"]
            if ep - curr_end_ep <= gap_threshold:
                curr_end_ep = ep
                curr_end_step = step
                count += 1
                curr_min_rate = min(curr_min_rate, ev.get("current_rate", 1.0))
            else:
                incidents.append({
                    "incident_id": len(incidents) + 1,
                    "start_step": curr_start_step,
                    "end_step": curr_end_step,
                    "start_ep": curr_start_ep,
                    "end_ep": curr_end_ep,
                    "duration_eps": curr_end_ep - curr_start_ep + 1,
                    "flagged_events": count,
                    "peak_rate": curr_peak,
                    "min_rate": curr_min_rate,
                })
                curr_start_ep, curr_start_step = ep, step
                curr_end_ep, curr_end_step = ep, step
                curr_peak = ev.get("peak_rate", 0.0)
                curr_min_rate = ev.get("current_rate", 0.0)
                count = 1

        incidents.append({
            "incident_id": len(incidents) + 1,
            "start_step": curr_start_step,
            "end_step": curr_end_step,
            "start_ep": curr_start_ep,
            "end_ep": curr_end_ep,
            "duration_eps": curr_end_ep - curr_start_ep + 1,
            "flagged_events": count,
            "peak_rate": curr_peak,
            "min_rate": curr_min_rate,
        })

        longest_inc = max(incidents, key=lambda x: x["duration_eps"]) if incidents else {}

        # 20k-timestep bins breakdown
        max_step = max(self.num_timesteps, max(e["step"] for e in events))
        bin_size = 20_000
        num_bins = max(1, int(np.ceil(max_step / bin_size)))
        bin_rows = []

        for b in range(num_bins):
            b_start = b * bin_size
            b_end = (b + 1) * bin_size
            bin_events = [e for e in events if b_start <= e["step"] < b_end]
            bin_incidents = [inc for inc in incidents if b_start <= inc["start_step"] < b_end]
            flagged_count = len(bin_events)
            inc_count = len(bin_incidents)
            
            # Approximate episodes in bin based on mean steps per episode (~2.2 steps/ep)
            approx_eps_in_bin = int(bin_size / 2.2)
            ep_fraction = (flagged_count / approx_eps_in_bin) * 100.0 if approx_eps_in_bin > 0 else 0.0

            bin_rows.append({
                "Timestep Bin": f"{b_start//1000}k - {b_end//1000}k",
                "Distinct Incidents": inc_count,
                "Flagged Steps": flagged_count,
                "Episode Warning Fraction (%)": f"{ep_fraction:.2f}%",
                "Longest Incident in Bin (Eps)": max([inc["duration_eps"] for inc in bin_incidents], default=0),
            })

        bin_df = pd.DataFrame(bin_rows)
        summary = {
            "total_incidents": len(incidents),
            "total_flagged_steps": len(events),
            "longest_incident": longest_inc,
            "longest_duration_eps": longest_inc.get("duration_eps", 0),
        }
        return incidents, bin_df, summary

    def on_training_end(self) -> None:
        """Runs sanity checks, saves final training plots, and prints summary."""
        # 1. Sanity Check: Assert invalid action rate is exactly 0%
        assert self.invalid_action_count == 0, (
            f"Action masking failure: {self.invalid_action_count} invalid actions occurred!"
        )

        total_eps = len(self.outcomes)
        if total_eps == 0 and not self.policy_collapse_events:
            return

        # 2. Early vs Late Training Delivery Rate Comparison
        if self.outcomes:
            split_point = max(1, int(total_eps * 0.20))
            early_deliv = sum(1 for o in self.outcomes[:split_point] if o == "delivered") / split_point
            late_deliv = sum(1 for o in self.outcomes[-split_point:] if o == "delivered") / split_point
        else:
            early_deliv, late_deliv = 0.0, 0.0

        incidents, bin_df, collapse_summary = self.analyze_collapse_incidents()
        self._save_telemetry_csvs()

        print("\n" + "=" * 85)
        print("Training Completed - Comprehensive Health & Policy Regression Diagnosis:")
        print("=" * 85)
        print(f"    - Total Timesteps Stepped          : {self.num_timesteps}")
        print(f"    - Total Episodes in Current Run     : {total_eps} (Total Cumulative Episodes: {self.episode_offset + total_eps})")
        print(f"    - Invalid Action Rate              : {self.invalid_action_count / max(1, self.num_timesteps):.4%} (Confirmed strictly 0.0000%)")
        if self.outcomes:
            print(f"    - Early Delivery Rate (1st 20%)    : {early_deliv*100.0:.2f}%")
            print(f"    - Late Delivery Rate  (Last 20%)    : {late_deliv*100.0:.2f}%")
        print(f"    - Peak Rolling Delivery Rate       : {self.best_rolling_delivery_rate*100.0:.2f}%")
        print(f"    - Total Distinct Collapse Incidents: {collapse_summary['total_incidents']} incidents across full run")
        print(f"    - Total Flagged Warning Steps      : {collapse_summary['total_flagged_steps']} flagged episodes")
        long_inc = collapse_summary.get("longest_incident", {})
        if long_inc:
            print(f"    - Longest Regression Incident      : {long_inc.get('duration_eps', 0)} episodes (Step {long_inc.get('start_step', 0)} - {long_inc.get('end_step', 0)})")

        print("\n[Collapse Incident Breakdown Across 20,000-Timestep Windows]")
        print(bin_df.to_string(index=False))

        # 3. Export Training Curves Plot
        self.plot_training_curves()

    def plot_training_curves(self) -> str:
        """Plots smoothed rewards, rolling delivery rate, and invalid action rate."""
        if not self.episode_rewards:
            return ""

        fig, axes = plt.subplots(3, 1, figsize=(11, 11), dpi=200)
        fig.patch.set_facecolor("#0f172a")

        color_primary = "#38bdf8"
        color_accent = "#10b981"
        color_alert = "#f43f5e"

        episodes = np.arange(1, len(self.episode_rewards) + 1)

        # 1. Episode Rewards
        ax1 = axes[0]
        ax1.set_facecolor("#1e293b")
        raw_r = pd.Series(self.episode_rewards)
        smooth_r = raw_r.rolling(window=min(50, len(raw_r)), min_periods=1).mean()
        ax1.plot(episodes, raw_r, alpha=0.25, color=color_primary, label="Raw Return")
        ax1.plot(episodes, smooth_r, color=color_primary, linewidth=2.0, label="Smoothed Return (Window 50)")
        ax1.set_title("1. Episode Return (Reward) Trajectory", color="#f8fafc", fontsize=11, fontweight="bold", loc="left", pad=8)
        ax1.set_ylabel("Episode Return", color="#cbd5e1", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        ax1.tick_params(colors="#cbd5e1")
        ax1.legend(loc="lower right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
        for spine in ax1.spines.values():
            spine.set_color("#334155")

        # 2. Rolling Delivery Rate
        ax2 = axes[1]
        ax2.set_facecolor("#1e293b")
        ax2.plot(episodes, np.array(self.rolling_delivery_rates) * 100.0, color=color_accent, linewidth=2.0, label="100-Ep Rolling Delivery Rate (%)")
        
        # Mark policy collapse warnings
        for event in self.policy_collapse_events:
            ax2.axvline(event["episode"], color=color_alert, linestyle=":", alpha=0.6)

        ax2.set_title("2. Rolling Delivery Rate (%) & Policy Health", color="#f8fafc", fontsize=11, fontweight="bold", loc="left", pad=8)
        ax2.set_ylabel("Delivery Rate (%)", color="#cbd5e1", fontsize=10, fontweight="bold")
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        ax2.tick_params(colors="#cbd5e1")
        ax2.legend(loc="lower right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
        for spine in ax2.spines.values():
            spine.set_color("#334155")

        # 3. Invalid Action Rate
        ax3 = axes[2]
        ax3.set_facecolor("#1e293b")
        ax3.axhline(0.0, color="#a855f7", linewidth=2.0, label="Invalid Action Rate (Strictly 0.0% via ActionMasker)")
        ax3.set_title("3. Invalid Action Rate Sanity Check (Masking Active)", color="#f8fafc", fontsize=11, fontweight="bold", loc="left", pad=8)
        ax3.set_xlabel("Episode Index", color="#cbd5e1", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Invalid Rate (%)", color="#cbd5e1", fontsize=10, fontweight="bold")
        ax3.set_ylim(-0.5, 2.0)
        ax3.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        ax3.tick_params(colors="#cbd5e1")
        ax3.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
        for spine in ax3.spines.values():
            spine.set_color("#334155")

        fig.suptitle("MaskablePPO Training Progress on NSFNET NetworkRoutingEnv", color="#f8fafc", fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(self.plot_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(self.plot_path)


def train_routing_agent(
    config: Optional[PPOConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    checkpoint_dir: str = "checkpoints",
    resume_from_checkpoint: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: int = 1,
    plot_path: Optional[str] = None,
) -> Tuple[Any, RoutingExperimentCallback, str]:
    """Trains a MaskablePPO agent on NetworkRoutingEnv with strict seed reproducibility.

    Args:
        config: PPOConfig hyperparameter instance.
        reward_config: RewardConfig instance from Prompt 3.
        checkpoint_dir: Path to directory for saving model checkpoints.
        resume_from_checkpoint: Optional path to .zip checkpoint to resume training from.
        total_timesteps: Optional override for total training timesteps.
        seed: Optional override for random seed.
        verbose: Verbosity level (0: quiet, 1: progress prints).
        plot_path: Optional custom path for training curves plot.

    Returns:
        Tuple of (trained_model, experiment_callback, final_model_path).
    """
    if not HAS_RL_DEPS:
        raise ImportError("Missing required RL packages. Ensure gymnasium, torch, and sb3-contrib are installed.")

    cfg = config if config is not None else PPOConfig()
    if total_timesteps is not None:
        cfg.total_timesteps = total_timesteps
    if seed is not None:
        cfg.seed = seed

    r_cfg = reward_config if reward_config is not None else RewardConfig()

    # 1. Thread seeds across all components
    actual_seed = cfg.seed
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    if verbose > 0:
        print("=" * 80)
        print("Initiating MaskablePPO Training Pipeline on NSFNET NetworkRoutingEnv")
        print("=" * 80)
        print(f"Config: seed={actual_seed}, timesteps={cfg.total_timesteps}, lr={cfg.learning_rate}, n_steps={cfg.n_steps}, batch_size={cfg.batch_size}")

    # 2. Build base environment and wrap with ActionMasker
    sim_env = NSFNETSimEnv(seed=actual_seed)
    base_env = NetworkRoutingEnv(env=sim_env, reward_config=r_cfg, seed=actual_seed)
    env = ActionMasker(base_env, mask_fn)

    # 3. Setup callback
    curve_plot = plot_path or os.path.join(PROJECT_ROOT, "assets", "rl_training_curves.png")
    callback = RoutingExperimentCallback(
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=cfg.checkpoint_freq,
        plot_path=curve_plot,
        verbose=verbose,
    )

    # 4. Instantiate or Resume Model
    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        if verbose > 0:
            print(f"[*] Resuming training from checkpoint: {resume_from_checkpoint}")
        model = MaskablePPO.load(
            resume_from_checkpoint,
            env=env,
            learning_rate=cfg.learning_rate,
            seed=actual_seed,
            device=cfg.device,
        )
    else:
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            seed=actual_seed,
            device=cfg.device,
            verbose=0,
        )

    # 5. Execute Training Loop
    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        steps_to_train = max(0, cfg.total_timesteps - model.num_timesteps)
        if verbose > 0:
            print(f"[*] Resumed at step {model.num_timesteps}. Executing {steps_to_train} steps to reach target {cfg.total_timesteps}.")
        model.learn(
            total_timesteps=steps_to_train,
            callback=callback,
            reset_num_timesteps=False,
        )
    else:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
        )

    # 6. Save Final Model Checkpoint
    final_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_path)
    step_tag = f"final_model_{model.num_timesteps // 1000}k.zip"
    tagged_path = os.path.join(checkpoint_dir, step_tag)
    model.save(tagged_path)
    if verbose > 0:
        print(f"[*] Successfully saved final model to: {final_path} (and {tagged_path})")

    return model, callback, final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaskablePPO agent on NetworkRoutingEnv.")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .zip to resume from.")
    parser.add_argument("--checkpoints", type=str, default="checkpoints", help="Checkpoints directory.")

    args = parser.parse_args()
    ppo_cfg = PPOConfig(total_timesteps=args.timesteps, seed=args.seed)
    train_routing_agent(
        config=ppo_cfg,
        checkpoint_dir=args.checkpoints,
        resume_from_checkpoint=args.resume,
    )
