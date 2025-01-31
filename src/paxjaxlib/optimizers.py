# optimizers.py
from typing import List, Tuple
import jax.numpy as jnp
from jax import tree_util

class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def apply_gradients(self, params: List[Tuple], grads: List[Tuple]) -> List[Tuple]:
        raise NotImplementedError

class SGD(Optimizer):
    def apply_gradients(self, params, grads):
        return [(W - self.learning_rate * dW, b - self.learning_rate * db) 
                for (W, b), (dW, db) in zip(params, grads)]

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # Moments for (weights, biases)
        self.v = None
        self.t = 0

    def apply_gradients(self, params, grads):
        if self.m is None or self.v is None:
            # Initialize moments for each parameter (W and b)
            self.m = [ (jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params ]
            self.v = [ (jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params ]
        
        self.t += 1
        new_params, new_m, new_v = [], [], []
        
        for (W, b), (dW, db), (m_w, m_b), (v_w, v_b) in zip(params, grads, self.m, self.v):
            # Update moments for weights
            m_w_new = self.beta1 * m_w + (1 - self.beta1) * dW
            v_w_new = self.beta2 * v_w + (1 - self.beta2) * (dW ** 2)
            m_w_hat = m_w_new / (1 - self.beta1 ** self.t)
            v_w_hat = v_w_new / (1 - self.beta2 ** self.t)
            W_new = W - self.learning_rate * m_w_hat / (jnp.sqrt(v_w_hat) + self.eps)
            
            # Update moments for biases
            m_b_new = self.beta1 * m_b + (1 - self.beta1) * db
            v_b_new = self.beta2 * v_b + (1 - self.beta2) * (db ** 2)
            m_b_hat = m_b_new / (1 - self.beta1 ** self.t)
            v_b_hat = v_b_new / (1 - self.beta2 ** self.t)
            b_new = b - self.learning_rate * m_b_hat / (jnp.sqrt(v_b_hat) + self.eps)
            
            new_params.append((W_new, b_new))
            new_m.append((m_w_new, m_b_new))
            new_v.append((v_w_new, v_b_new))
        
        self.m, self.v = new_m, new_v
        return new_params
