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

class AdaMax(Optimizer):
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # Moments for (weights, biases)
        self.u = None  # Infinity norm for (weights, biases)
        self.t = 0

    def apply_gradients(self, params, grads):
        if self.m is None or self.u is None:
            self.m = [(jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params]
            self.u = [(jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params]
        
        self.t += 1
        new_params, new_m, new_u = [], [], []
        
        for (W, b), (dW, db), (m_w, m_b), (u_w, u_b) in zip(params, grads, self.m, self.u):
            # Update moments for weights
            m_w_new = self.beta1 * m_w + (1 - self.beta1) * dW
            u_w_new = jnp.maximum(self.beta2 * u_w, jnp.abs(dW))
            W_new = W - self.learning_rate * m_w_new / (u_w_new + self.eps)
            
            # Update moments for biases
            m_b_new = self.beta1 * m_b + (1 - self.beta1) * db
            u_b_new = jnp.maximum(self.beta2 * u_b, jnp.abs(db))
            b_new = b - self.learning_rate * m_b_new / (u_b_new + self.eps)
            
            new_params.append((W_new, b_new))
            new_m.append((m_w_new, m_b_new))
            new_u.append((u_w_new, u_b_new))
        
        self.m, self.u = new_m, new_u
        return new_params

class RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps
        self.cache = None  # Cache for squared gradients

    def apply_gradients(self, params, grads):
        if self.cache is None:
            self.cache = [(jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params]
        
        new_params, new_cache = [], []
        
        for (W, b), (dW, db), (cache_w, cache_b) in zip(params, grads, self.cache):
            # Update cache for weights
            cache_w_new = self.rho * cache_w + (1 - self.rho) * (dW ** 2)
            W_new = W - self.learning_rate * dW / (jnp.sqrt(cache_w_new) + self.eps)
            
            # Update cache for biases
            cache_b_new = self.rho * cache_b + (1 - self.rho) * (db ** 2)
            b_new = b - self.learning_rate * db / (jnp.sqrt(cache_b_new) + self.eps)
            
            new_params.append((W_new, b_new))
            new_cache.append((cache_w_new, cache_b_new))
        
        self.cache = new_cache
        return new_params

class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = None  # Velocity for (weights, biases)

    def apply_gradients(self, params, grads):
        if self.velocity is None:
            self.velocity = [(jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params]
        
        new_params, new_velocity = [], []
        
        for (W, b), (dW, db), (v_w, v_b) in zip(params, grads, self.velocity):
            # Update velocity for weights
            v_w_new = self.momentum * v_w - self.learning_rate * dW
            W_new = W + v_w_new
            
            # Update velocity for biases
            v_b_new = self.momentum * v_b - self.learning_rate * db
            b_new = b + v_b_new
            
            new_params.append((W_new, b_new))
            new_velocity.append((v_w_new, v_b_new))
        
        self.velocity = new_velocity
        return new_params

class Adafactor(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-30):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # Moments for (weights, biases)
        self.v = None  # Second moments for (weights, biases)
        self.t = 0

    def apply_gradients(self, params, grads):
        if self.m is None or self.v is None:
            self.m = [(jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params]
            self.v = [(jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params]
        
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

class AdaGrad(Optimizer):
    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.cache = None  # Cache for squared gradients

    def apply_gradients(self, params, grads):
        if self.cache is None:
            self.cache = [(jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params]
        
        new_params, new_cache = [], []
        
        for (W, b), (dW, db), (cache_w, cache_b) in zip(params, grads, self.cache):
            # Update cache for weights
            cache_w_new = cache_w + dW ** 2
            W_new = W - self.learning_rate * dW / (jnp.sqrt(cache_w_new) + self.eps)
            
            # Update cache for biases
            cache_b_new = cache_b + db ** 2
            b_new = b - self.learning_rate * db / (jnp.sqrt(cache_b_new) + self.eps)
            
            new_params.append((W_new, b_new))
            new_cache.append((cache_w_new, cache_b_new))
        
        self.cache = new_cache
        return new_params

class Adadelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-8):
        super().__init__(1.0)  # Adadelta doesn't use learning rate
        self.rho = rho
        self.eps = eps
        self.acc_grad = None  # Accumulated gradients
        self.acc_update = None  # Accumulated updates

    def apply_gradients(self, params, grads):
        if self.acc_grad is None or self.acc_update is None:
            self.acc_grad = [(jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params]
            self.acc_update = [(jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in params]
        
        new_params, new_acc_grad, new_acc_update = [], [], []
        
        for (W, b), (dW, db), (acc_grad_w, acc_grad_b), (acc_update_w, acc_update_b) in zip(params, grads, self.acc_grad, self.acc_update):
            # Update accumulated gradients for weights
            acc_grad_w_new = self.rho * acc_grad_w + (1 - self.rho) * (dW ** 2)
            update_w = -jnp.sqrt(acc_update_w + self.eps) / jnp.sqrt(acc_grad_w_new + self.eps) * dW
            W_new = W + update_w
            acc_update_w_new = self.rho * acc_update_w + (1 - self.rho) * (update_w ** 2)
            
            # Update accumulated gradients for biases
            acc_grad_b_new = self.rho * acc_grad_b + (1 - self.rho) * (db ** 2)
            update_b = -jnp.sqrt(acc_update_b + self.eps) / jnp.sqrt(acc_grad_b_new + self.eps) * db
            b_new = b + update_b
            acc_update_b_new = self.rho * acc_update_b + (1 - self.rho) * (update_b ** 2)
            
            new_params.append((W_new, b_new))
            new_acc_grad.append((acc_grad_w_new, acc_grad_b_new))
            new_acc_update.append((acc_update_w_new, acc_update_b_new))
        
        self.acc_grad, self.acc_update = new_acc_grad, new_acc_update
        return new_params


