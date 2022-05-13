import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .utils import topological_order

__all__ = ["Value",]

@dataclass
class Value:
    value: np.ndarray = None
    gradient: Optional[np.ndarray] = None
    requires_gradient: bool = True
    parents = ()

    @classmethod
    def wrap_value(cls, value):
        if not isinstance(value, Value):
            value = Value(value)
        
        return value

    def clear_gradient(self):
        self.gradient = None

    def receive_gradient(self, gradient):
        if not self.requires_gradient: 
            return

        if self.gradient is None:
            self.gradient = gradient
            return

        self.gradient += gradient

    def pass_gradients(self):
        pass

    def backward(self):
        order = topological_order(self, lambda node: node.parents)

        self.gradient = 1
        for n in order:
            n.pass_gradients()

    def __add__(self, other):
        from .ops import Add
        return Add(self, Value.wrap_value(other))
    
    def __mul__(self, other):
        from .ops import Multiply
        return Multiply(self, Value.wrap_value(other))
    
    def __matmul__(self, other):
        from .ops import MatrixMultiply
        return MatrixMultiply(self, Value.wrap_value(other))
    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return self * -1
    
    def __pow__(self, power):
        from .ops import Power
        return Power(self, Value.wrap_value(power))

    def sum(self):
        from .ops import Sum
        return Sum(self)

