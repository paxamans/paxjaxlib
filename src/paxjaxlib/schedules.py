"""
Learning-rate schedule functions for paxjaxlib.

Each function returns a callable ``scheduler(step) → lr`` that can be
composed with ``optax.schedule``-based optimisers.
"""


def exponential_decay(initial_lr: float, decay_rate: float, decay_steps: int):
    """Exponential learning-rate decay.

    ``lr(step) = initial_lr * decay_rate ^ (step / decay_steps)``

    Args:
        initial_lr: Starting learning rate.
        decay_rate: Multiplicative decay factor.
        decay_steps: Number of steps for one full decay cycle.
    """

    def scheduler(step: int) -> float:
        return float(initial_lr * decay_rate ** (step / decay_steps))

    return scheduler


def step_decay(initial_lr: float, drop_rate: float, steps_drop: int):
    """Step-wise learning-rate decay.

    Drops the learning rate by ``drop_rate`` every ``steps_drop`` steps.

    Args:
        initial_lr: Starting learning rate.
        drop_rate: Multiplicative factor applied at each drop.
        steps_drop: Number of steps between each drop.
    """

    def scheduler(step: int) -> float:
        return float(initial_lr * drop_rate ** (step // steps_drop))

    return scheduler
