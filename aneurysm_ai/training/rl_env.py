import numpy as np


class AneurysmSearchEnv:

    def __init__(
        self,
        volume_shape,
        aneurysm_centers,
        initial_position=None,
        initial_scale=1.0,
        move_step=0.08,
        scale_step=0.15,
        max_steps=64,
        distance_scale=1.0,
        confidence_weight=0.5,
        step_penalty=0.01,
    ):

        self.volume_shape = np.asarray(
            volume_shape,
            dtype=np.float32,
        )

        self.aneurysm_centers = np.asarray(
            aneurysm_centers,
            dtype=np.float32,
        )

        self.move_step = move_step
        self.scale_step = scale_step
        self.max_steps = max_steps

        self.distance_scale = (
            distance_scale
        )

        self.confidence_weight = (
            confidence_weight
        )

        self.step_penalty = (
            step_penalty
        )

        self.initial_position = (
            initial_position
        )

        self.initial_scale = (
            initial_scale
        )

        self.position = None
        self.scale = None
        self.confidence = None
        self.steps = 0

        self.previous_distance = None
        self.previous_confidence = None

    # --------------------------------------------------
    # Normalized position
    # --------------------------------------------------

    def _normalize_position(
        self,
        position,
    ):

        z, y, x = self.volume_shape

        return np.array([
            position[0] / max(x - 1, 1),
            position[1] / max(y - 1, 1),
            position[2] / max(z - 1, 1),
        ], dtype=np.float32)

    # --------------------------------------------------
    # Denormalized position
    # --------------------------------------------------

    def _denormalize_position(
        self,
        normalized,
    ):

        z, y, x = self.volume_shape

        return np.array([
            normalized[0] * (x - 1),
            normalized[1] * (y - 1),
            normalized[2] * (z - 1),
        ], dtype=np.float32)

    # --------------------------------------------------
    # Minimum distance to any aneurysm
    # --------------------------------------------------

    def _distance_to_target(self):

        distances = np.linalg.norm(
            self.aneurysm_centers
            - self.position[None, :],
            axis=1,
        )

        return float(
            np.min(distances)
        )

    # --------------------------------------------------
    # State
    # --------------------------------------------------

    def _state(self):

        position = self._normalize_position(
            self.position
        )

        return np.array([
            position[0],
            position[1],
            position[2],
            self.scale,
            self.confidence,
        ], dtype=np.float32)

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(
        self,
        initial_position=None,
        confidence=0.0,
    ):

        self.steps = 0

        if initial_position is None:

            if self.initial_position is not None:
                initial_position = (
                    self.initial_position
                )

            else:

                initial_position = (
                    self.volume_shape[
                        ::-1
                    ] - 1
                ) * 0.5

        self.position = np.asarray(
            initial_position,
            dtype=np.float32,
        )

        self.scale = float(
            self.initial_scale
        )

        self.confidence = float(
            confidence
        )

        self.previous_distance = (
            self._distance_to_target()
        )

        self.previous_confidence = (
            self.confidence
        )

        return self._state()

    # --------------------------------------------------
    # Apply action
    # --------------------------------------------------

    def _apply_action(
        self,
        action,
    ):

        if action == 0:
            self.position[0] += self.move_step * (
                self.volume_shape[2]
            )

        elif action == 1:
            self.position[0] -= self.move_step * (
                self.volume_shape[2]
            )

        elif action == 2:
            self.position[1] += self.move_step * (
                self.volume_shape[1]
            )

        elif action == 3:
            self.position[1] -= self.move_step * (
                self.volume_shape[1]
            )

        elif action == 4:
            self.position[2] += self.move_step * (
                self.volume_shape[0]
            )

        elif action == 5:
            self.position[2] -= self.move_step * (
                self.volume_shape[0]
            )

        elif action == 6:
            self.scale += self.scale_step

        elif action == 7:
            self.scale -= self.scale_step

        self.position[0] = np.clip(
            self.position[0],
            0,
            self.volume_shape[2] - 1,
        )

        self.position[1] = np.clip(
            self.position[1],
            0,
            self.volume_shape[1] - 1,
        )

        self.position[2] = np.clip(
            self.position[2],
            0,
            self.volume_shape[0] - 1,
        )

        self.scale = np.clip(
            self.scale,
            0.25,
            2.0,
        )

    # --------------------------------------------------
    # Step
    # --------------------------------------------------

    def step(
        self,
        action,
        new_confidence,
    ):

        self.steps += 1

        old_distance = (
            self.previous_distance
        )

        old_confidence = (
            self.previous_confidence
        )

        self._apply_action(action)

        new_distance = (
            self._distance_to_target()
        )

        new_confidence = float(
            new_confidence
        )

        distance_reward = (
            old_distance - new_distance
        ) * self.distance_scale

        confidence_reward = (
            new_confidence
            - old_confidence
        ) * self.confidence_weight

        reward = (
            distance_reward
            + confidence_reward
            - self.step_penalty
        )

        done = (
            action == 8
            or self.steps >= self.max_steps
        )

        self.previous_distance = (
            new_distance
        )

        self.previous_confidence = (
            new_confidence
        )

        self.confidence = (
            new_confidence
        )

        return (
            self._state(),
            float(reward),
            done,
            {
                "distance": new_distance,
                "confidence": new_confidence,
            },
        )