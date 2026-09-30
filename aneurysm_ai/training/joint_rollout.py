import torch


def collect_joint_rollout(
    env,
    adapter,
    agent,
    max_steps,
    device,
):
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []

    state = env.reset()

    for _ in range(max_steps):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():

            distribution, value = agent.distribution(
                state_tensor
            )

            action = distribution.sample()

            log_prob = distribution.log_prob(
                action
            )

        action_id = int(action.item())

        confidence = adapter.confidence(
            env.position,
            env.scale,
        )

        next_state, reward, done, info = env.step(
            action_id,
            confidence=confidence,
        )

        states.append(state)
        actions.append(action_id)
        rewards.append(reward)
        log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))
        dones.append(done)

        state = next_state

        if done:
            break

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs,
        "values": values,
        "dones": dones,
    }