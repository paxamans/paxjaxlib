from typing import List, Tuple
import jax.numpy as jnp
from jax import tree_util as tree

class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def apply_gradients(self, params: List[Tuple[jnp.ndarray, jnp.ndarray]], grads: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        raise NotImplementedError

class SGD(Optimizer):
    def apply_gradients(self, params: List[Tuple[jnp.ndarray, jnp.ndarray]], grads: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        return [(W - self.learning_rate * dW, b - self.learning_rate * db) 
                for (W, b), (dW, db) in zip(params, grads)]

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # first moment estimate
        self.v = None  # second moment estimate
        self.t = 0     # time step

    def apply_gradients(self, params: List[Tuple[jnp.ndarray, jnp.ndarray]], grads: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        if self.m is None or self.v is None:
            self.m = tree.tree_map(jnp.zeros_like, params)
            self.v = tree.tree_map(jnp.zeros_like, params)
        
        self.t += 1
        self.m = tree.tree_map(lambda m, g: self.beta1 * m + (1 - self.beta1) * g, self.m, grads)
        self.v = tree.tree_map(lambda v, g: self.beta2 * v + (1 - self.beta2) * (g ** 2), self.v, grads)
        
        m_hat = tree.tree_map(lambda m: m / (1 - self.beta1 ** self.t), self.m)
        v_hat = tree.tree_map(lambda v: v / (1 - self.beta2 ** self.t), self.v)
        
        return tree.tree_map(lambda p, m, v: p - self.learning_rate * m / (jnp.sqrt(v) + self.eps), params, m_hat, v_hat)
        
class AdaMax(Optimizer):
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # first moment estimate
        self.u = None  # weighted infinity norm
        self.t = 0

    def apply_gradients(self, params, grads):
        if self.m is None or self.u is None:
            self.m = tree.tree_map(jnp.zeros_like, params)
            self.u = tree.tree_map(jnp.zeros_like, params)
        
        self.t += 1
        # update moments
        self.m = tree.tree_map(lambda m, g: self.beta1 * m + (1 - self.beta1) * g, 
                             self.m, grads)
        self.u = tree.tree_map(lambda u, g: jnp.maximum(self.beta2 * u, jnp.abs(g)), 
                             self.u, grads)
        
        # bias correction
        m_hat = tree.tree_map(lambda m: m / (1 - self.beta1 ** self.t), self.m)
        
        return tree.tree_map(
            lambda p, m, u: p - self.learning_rate * m / (u + self.eps),
            params, m_hat, self.u
        )

class RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps
        self.cache = None  # moving average of squared gradients

    def apply_gradients(self, params, grads):
        if self.cache is None:
            self.cache = tree.tree_map(jnp.zeros_like, params)
        
        # update cache
        self.cache = tree.tree_map(
            lambda c, g: self.rho * c + (1 - self.rho) * jnp.square(g),
            self.cache, grads
        )
        
        return tree.tree_map(
            lambda p, g, c: p - self.learning_rate * g / (jnp.sqrt(c) + self.eps),
            params, grads, self.cache
        )

class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = None

    def apply_gradients(self, params, grads):
        if self.velocity is None:
            self.velocity = tree.tree_map(jnp.zeros_like, params)
        
        # update velocity
        self.velocity = tree.tree_map(
            lambda v, g: self.momentum * v - self.learning_rate * g,
            self.velocity, grads
        )
        
        return tree.tree_map(jnp.add, params, self.velocity)

class Adafactor(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-30):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # first moment estimate
        self.v = None  # second moment estimate
        self.t = 0

    def apply_gradients(self, params, grads):
        if self.m is None or self.v is None:
            self.m = tree.tree_map(jnp.zeros_like, params)
            self.v = tree.tree_map(jnp.zeros_like, params)
        
        self.t += 1
        self.m = tree.tree_map(lambda m, g: self.beta1 * m + (1 - self.beta1) * g, self.m, grads)
        self.v = tree.tree_map(lambda v, g: self.beta2 * v + (1 - self.beta2) * jnp.square(g), self.v, grads)
        
        m_hat = tree.tree_map(lambda m: m / (1 - self.beta1 ** self.t), self.m)
        v_hat = tree.tree_map(lambda v: v / (1 - self.beta2 ** self.t), self.v)
        
        return tree.tree_map(
            lambda p, m, v: p - self.learning_rate * m / (jnp.sqrt(v) + self.eps),
            params, m_hat, v_hat
        )

class AdaGrad(Optimizer):
    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.cache = None  # cache for squared gradients

    def apply_gradients(self, params, grads):
        if self.cache is None:
            self.cache = tree.tree_map(jnp.zeros_like, params)
        
        # update cache
        self.cache = tree.tree_map(lambda c, g: c + jnp.square(g), self.cache, grads)
        
        return tree.tree_map(
            lambda p, g, c: p - self.learning_rate * g / (jnp.sqrt(c) + self.eps),
            params, grads, self.cache
        )

class Adadelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-8):
        super().__init__(1.0)  # learning rate not used in Adadelta
        self.rho = rho
        self.eps = eps
        self.acc_grad = None  # accumulated gradients
        self.acc_update = None  # accumulated updates

    def apply_gradients(self, params, grads):
        if self.acc_grad is None or self.acc_update is None:
            self.acc_grad = tree.tree_map(jnp.zeros_like, params)
            self.acc_update = tree.tree_map(jnp.zeros_like, params)
        
        # update accumulated gradients
        new_acc_grad = tree.tree_map(
            lambda acc, g: self.rho * acc + (1 - self.rho) * jnp.square(g),
            self.acc_grad, grads
        )
    
        # Corrected updates calculation:
        updates = tree.tree_map(
            lambda acc_up, curr_acc_g, g: -(jnp.sqrt(acc_up + self.eps) / jnp.sqrt(curr_acc_g + self.eps)) * g,
            self.acc_update, new_acc_grad, grads  
        )
        
        # update parameters and accumulated updates
        new_params = tree.tree_map(jnp.add, params, updates)
        new_acc_update = tree.tree_map(
            lambda acc, upd: self.rho * acc + (1 - self.rho) * jnp.square(upd),
            self.acc_update, updates
        )
        
        self.acc_grad = new_acc_grad
        self.acc_update = new_acc_update
        
        return new_params
