import numpy as np

from .value import Value

__all__ = [
    "Add", "Subtract", "Multiply", "Power", "ReLU"
]

class Add(Value):
    def __init__(self, left: Value, right: Value):
        self.value = left.value + right.value
        self.parents = [left, right]

    def pass_gradients(self):
        self.parents[0].receive_gradient(self.gradient)
        self.parents[1].receive_gradient(self.gradient)

class Subtract(Value):
    def __init__(self, left: Value, right: Value):
        self.value = left.value - right.value
        self.parents = [left, right]
    
    def pass_gradients(self):
        self.parents[0].receive_gradient(self.gradient)
        self.parents[1].receive_gradient(-self.gradient)

class MatrixMultiply(Value):
    def __init__(self, a: Value, b: Value):
        self.value = a.value @ b.value
        self.parents = [a, b]
    
    def pass_gradients(self):
        a, b = self.parents
        if a.value.ndim == 1:
            b.receive_gradient(np.outer(a.value, self.gradient))
        else:
            b.receive_gradient(a.value.T @ self.gradient)

        if b.value.ndim == 1:
            a.receive_gradient(np.outer(self.gradient, b.value))
        else:
            a.receive_gradient(self.gradient @ b.value.T)

class Multiply(Value):
    def __init__(self, left: Value, right: Value):
        self.value = left.value * right.value
        self.parents = [left, right]
    
    def pass_gradients(self):
        self.parents[0].receive_gradient(self.gradient * self.parents[1].value)
        self.parents[1].receive_gradient(self.gradient * self.parents[0].value)

class Power(Value):
    def __init__(self, left: Value, right: Value):
        self.value = left.value ** right.value
        self.parents = [left, right]
    
    def pass_gradients(self):
        l, r = self.parents
        g = self.gradient
        l.receive_gradient(g * self.value * r.value / l.value)
        r.receive_gradient(g * self.value * np.log(l.value))

class ReLU(Value):
    def __init__(self, x):
        self.value = np.clip(x, 0, None)
        self.parents = [x]
    
    def pass_gradients(self):
        mask = (np.sign(self.value) + 1) / 2
        self.parents[0].receive_gradient(mask * self.gradient)

class Sum(Value):
    def __init__(self, x):
        self.value = x.value.sum()
        self.parents = [x]
    
    def pass_gradients(self):
        self.parents[0].receive_gradient(
            self.gradient * np.ones_like(self.parents[0].value)
        )