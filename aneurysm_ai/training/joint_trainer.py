from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .aneurysm_trainer import AneurysmDetectionLoss
from .ppo_trainer import PPOTrainer
from .joint_rollout import collect_joint_rollout


@dataclass
class JointConfig:
    num_episodes: int = 32
    lambda_rl: float = 0.01

    detector_lr: float = 1e-5
    ppo_lr: float = 3e-4

    grad_clip: float = 1.0

    checkpoint_dir: str = "checkpoints"


class JointTrainer:
    """
    Alternating supervised + PPO training.

    Detector:
        optimized using supervised detection loss.

    PPO:
        optimized from search trajectories.

    The environment is intentionally kept outside the autograd graph.
    """

    def __init__(
        self,
        detector,
        agent,
        detector_loss=None,
        config=None,
        device="cuda",
    ):
        self.detector = detector
        self.agent = agent

        self.cfg = config or JointConfig()
        self.device = torch.device(device)

        self.detector.to(self.device)
        self.agent.to(self.device)

        self.detector_loss = detector_loss or AneurysmDetectionLoss()

        self.detector_optimizer = torch.optim.AdamW(
            self.detector.parameters(),
            lr=self.cfg.detector_lr,
            weight_decay=1e-4,
        )

        self.ppo_optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=self.cfg.ppo_lr,
            weight_decay=1e-4,
        )

        self.ppo = PPOTrainer(
            self.agent,
            optimizer=self.ppo_optimizer,
        )

        Path(self.cfg.checkpoint_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

    def supervised_step(self, batch):
        self.detector.train()

        images = batch["image"].to(self.device)
        objectness = batch["objectness"].to(self.device)
        center = batch["center"].to(self.device)
        anatomy = batch["anatomy"].to(self.device)

        predictions = self.detector(images)

        losses = self.detector_loss(
            predictions,
            objectness,
            center,
            anatomy,
        )

        loss = losses["total"]

        self.detector_optimizer.zero_grad(set_to_none=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.detector.parameters(),
            self.cfg.grad_clip,
        )

        self.detector_optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "objectness": float(
                losses["objectness"].detach().cpu()
            ),
            "center": float(
                losses["center"].detach().cpu()
            ),
            "anatomy": float(
                losses["anatomy"].detach().cpu()
            ),
        }

    @torch.no_grad()
    def detector_confidence(self, images):
        self.detector.eval()

        predictions = self.detector(images)

        confidence = torch.sigmoid(
            predictions["objectness"]
        )

        return confidence

    def save_checkpoint(self, epoch, metrics):
        path = (
            Path(self.cfg.checkpoint_dir)
            / f"joint_epoch_{epoch:03d}.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "detector": self.detector.state_dict(),
                "agent": self.agent.state_dict(),
                "detector_optimizer":
                    self.detector_optimizer.state_dict(),
                "ppo_optimizer":
                    self.ppo_optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )

        return path
    def train_epoch(
        self,
        dataloader,
        environments,
        adapter_factory,
        max_steps=64,
    ):
        supervised_metrics = []

        # -----------------------------
        # 1. Supervised detector update
        # -----------------------------

        for batch in dataloader:

            metrics = self.supervised_step(batch)

            supervised_metrics.append(metrics)

        # -----------------------------
        # 2. PPO episodes
        # -----------------------------

        episode_returns = []
        episode_lengths = []

        self.agent.train()

        for episode_id in range(
            self.cfg.num_episodes
        ):

            env = environments[
                episode_id % len(environments)
            ]

            adapter = adapter_factory(env)

            rollout = collect_joint_rollout(
                env=env,
                adapter=adapter,
                agent=self.agent,
                max_steps=max_steps,
                device=self.device,
            )

            rewards = rollout["rewards"]

            episode_returns.append(
                sum(rewards)
            )

            episode_lengths.append(
                len(rewards)
            )

            # Convert trajectory for PPO update.
            self.ppo.update(rollout)

        # -----------------------------
        # 3. Aggregate
        # -----------------------------

        mean_supervised = (
            sum(x["loss"] for x in supervised_metrics)
            / max(len(supervised_metrics), 1)
        )

        mean_return = (
            sum(episode_returns)
            / max(len(episode_returns), 1)
        )

        mean_length = (
            sum(episode_lengths)
            / max(len(episode_lengths), 1)
        )

        return {
            "supervised_loss": mean_supervised,
            "ppo_return": mean_return,
            "episode_length": mean_length,
        }