from typing import Iterable, Mapping, Sequence


def success_rate(episodes: Iterable[Mapping[str, object]]) -> float:
    items = list(episodes)
    if not items:
        return 0.0
    successes = sum(1 for episode in items if bool(episode.get("success", False)))
    return successes / len(items)


def action_mse(predicted: Sequence[Sequence[float]], target: Sequence[Sequence[float]]) -> float:
    if len(predicted) != len(target):
        raise ValueError("predicted and target must have the same number of timesteps.")

    total = 0.0
    count = 0
    for predicted_step, target_step in zip(predicted, target):
        if len(predicted_step) != len(target_step):
            raise ValueError("Every predicted action must match the target action dimension.")
        for predicted_value, target_value in zip(predicted_step, target_step):
            error = float(predicted_value) - float(target_value)
            total += error * error
            count += 1

    if count == 0:
        return 0.0
    return total / count
