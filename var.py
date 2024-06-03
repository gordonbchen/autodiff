from __future__ import annotations

import math

from typing import Callable


def force_var(func: Callable[[Var, Var | float], Var]) -> Callable[[Var, Var], Var]:
    """Decorator to convert float to a Var if necessary."""

    def var_func(this: Var, other: Var | float) -> Var:
        other = other if type(other) is Var else Var(other)
        return func(this, other)

    return var_func


class Var:
    """A differentiable variable with a float value."""

    def __init__(
        self,
        val: float,
        parents: tuple[Var] = tuple(),
        parent_grads: tuple[float] = tuple(),
    ) -> None:
        """Initialize variable with value, parents, and parent grads."""
        self.val = val
        self.parents = parents
        self.parent_grads = parent_grads

        self.grad = 0.0  # Gradient will be accumulated on backward() call.

    def backward(self, child_grad: float = 1.0) -> None:
        """Backpropagate gradients."""
        self.grad += child_grad

        for parent, parent_grad in zip(self.parents, self.parent_grads):
            parent.backward(child_grad=parent_grad * child_grad)  # Chain rule.

    # Fundamental ops.
    @force_var
    def __add__(self, other: Var | float) -> Var:
        return Var(
            val=self.val + other.val, parents=(self, other), parent_grads=(1.0, 1.0)
        )

    @force_var
    def __mul__(self, other: Var | float) -> Var:
        return Var(
            val=self.val * other.val,
            parents=(self, other),
            parent_grads=(other.val, self.val),
        )

    @force_var
    def __pow__(self, other: Var | float) -> Var:
        this_grad = other.val * (self.val ** (other.val - 1))
        other_grad = (self.val**other.val) * math.log(self.val, math.e)
        return Var(
            val=self.val**other.val,
            parents=(self, other),
            parent_grads=(this_grad, other_grad),
        )

    # Derived ops.
    def __neg__(self) -> Var:
        return self * -1.0

    @force_var
    def __sub__(self, other: Var | float) -> Var:
        return self + (-other)

    @force_var
    def __truediv__(self, other: Var | float) -> Var:
        return self * (other**-1.0)

    # Right hand ops. @force_var converts other to Var, no modifications needed.
    @force_var
    def __radd__(self, other: Var | float) -> Var:
        return other + self

    @force_var
    def __rsub__(self, other: Var | float) -> Var:
        return other - self

    @force_var
    def __rmul__(self, other: Var | float) -> Var:
        return other * self

    @force_var
    def __rtruediv__(self, other: Var | float) -> Var:
        return other / self

    @force_var
    def __rpow__(self, other: Var | float) -> Var:
        return other**self

    # Other math ops.
    def ln(self) -> Var:
        return Var(
            val=math.log(self.val, math.e),
            parents=(self,),
            parent_grads=(1.0 / self.val,),
        )

    # NN ops.
    def relu(self) -> Var:
        return Var(
            val=self.val if (self.val > 0.0) else 0.0,
            parents=(self,),
            parent_grads=(1.0 if (self.val > 0.0) else 0.0,),
        )
