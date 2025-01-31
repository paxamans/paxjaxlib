# schedules.py
import jax.numpy as jnp

def exponential_decay(initial_lr: float, decay_rate: float, decay_steps: int):
    def scheduler(step):
        return initial_lr * decay_rate ** (step / decay_steps)
    return scheduler

def step_decay(initial_lr: float, drop_rate: float, epochs_drop: int):
    def scheduler(epoch):
        return initial_lr * drop_rate ** (epoch // epochs_drop)
    return scheduler
