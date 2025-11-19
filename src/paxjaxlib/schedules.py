# schedules.py


def exponential_decay(initial_lr: float, decay_rate: float, decay_steps: int):
    def scheduler(step):
        return initial_lr * decay_rate ** (step / decay_steps)

    return scheduler


def step_decay(
    initial_lr: float, drop_rate: float, steps_drop: int
):  # Renamed epochs_drop
    def scheduler(step):  # Takes step instead of epoch
        return initial_lr * drop_rate ** (step // steps_drop)

    return scheduler
