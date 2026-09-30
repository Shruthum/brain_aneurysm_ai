import torch
import torch.nn.functional as F


class PPOTrainer:

    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        epochs=4,
    ):

        self.model = model

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.epochs = epochs

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

    # --------------------------------------------------
    # Action selection
    # --------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        state,
    ):

        distribution, value = (
            self.model.distribution(
                state
            )
        )

        action = distribution.sample()

        log_probability = (
            distribution.log_prob(action)
        )

        return (
            action,
            log_probability,
            value,
        )

    # --------------------------------------------------
    # GAE
    # --------------------------------------------------

    def compute_advantages(
        self,
        rewards,
        values,
        dones,
        next_value,
    ):

        advantages = []

        gae = torch.zeros(
            (),
            device=rewards.device,
        )

        values = torch.cat(
            [
                values,
                next_value.unsqueeze(0),
            ]
        )

        for t in reversed(
            range(len(rewards))
        ):

            mask = 1.0 - dones[t]

            delta = (
                rewards[t]
                + self.gamma
                * values[t + 1]
                * mask
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * mask
                * gae
            )

            advantages.insert(
                0,
                gae,
            )

        return torch.stack(
            advantages
        )

    # --------------------------------------------------
    # PPO update
    # --------------------------------------------------

    def update(
        self,
        states,
        actions,
        old_log_probs,
        returns,
        advantages,
    ):

        advantages = (
            advantages
            - advantages.mean()
        ) / (
            advantages.std()
            + 1e-8
        )

        for _ in range(self.epochs):

            distribution, values = (
                self.model.distribution(
                    states
                )
            )

            new_log_probs = (
                distribution.log_prob(
                    actions
                )
            )

            entropy = (
                distribution.entropy()
            ).mean()

            ratio = torch.exp(
                new_log_probs
                - old_log_probs
            )

            unclipped = (
                ratio * advantages
            )

            clipped = torch.clamp(
                ratio,
                1.0 - self.clip_eps,
                1.0 + self.clip_eps,
            ) * advantages

            policy_loss = -torch.min(
                unclipped,
                clipped,
            ).mean()

            value_loss = F.mse_loss(
                values,
                returns,
            )

            loss = (
                policy_loss
                + self.value_coef
                * value_loss
                - self.entropy_coef
                * entropy
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0,
            )

            self.optimizer.step()

        return {
            "loss": float(loss.detach()),
            "policy_loss": float(
                policy_loss.detach()
            ),
            "value_loss": float(
                value_loss.detach()
            ),
            "entropy": float(
                entropy.detach()
            ),
        }

def collect_rollout(
    env,
    agent,
    trainer,
    device,
    detector_confidence_fn,
    rollout_steps=64,
):

    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []

    state = env.reset(
        confidence=0.0
    )

    for _ in range(
        rollout_steps
    ):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        action, log_prob, value = (
            trainer.select_action(
                state_tensor
            )
        )

        action_int = int(
            action.item()
        )

        # ----------------------------------------------
        # Detector evaluates the new candidate.
        # ----------------------------------------------

        new_confidence = (
            detector_confidence_fn(
                env.position,
                env.scale,
            )
        )

        next_state, reward, done, info = (
            env.step(
                action_int,
                new_confidence,
            )
        )

        states.append(state_tensor.squeeze(0))
        actions.append(action.squeeze(0))
        rewards.append(reward)
        log_probs.append(
            log_prob.squeeze(0)
        )
        values.append(
            value.squeeze(0)
        )
        dones.append(
            float(done)
        )

        state = next_state

        if done:
            break

    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(
        log_probs
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device,
    )

    values = torch.stack(values)

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        next_state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        _, next_value = (
            agent.distribution(
                next_state_tensor
            )
        )

    advantages = trainer.compute_advantages(
        rewards,
        values,
        dones,
        next_value.squeeze(0),
    )

    returns = (
        advantages + values
    )

    return {
        "states": states,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "advantages": advantages.detach(),
        "returns": returns.detach(),
    }