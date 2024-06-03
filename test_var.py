import math
import torch

from var import Var


def test_relu_linear():
    a = Var(1.0)
    b = a.relu()
    b.backward()

    assert b.val == 1.0
    assert a.grad == 1.0


def test_relu_zero():
    a = Var(-1.0)
    b = a.relu()
    b.backward()

    assert b.val == 0.0
    assert a.grad == 0.0


def test_ln():
    a = Var(10.0)
    b = a.ln()
    b.backward()

    assert math.isclose(b.val, math.log(a.val, math.e))
    assert a.grad == (1.0 / 10.0)


def test_add():
    a = Var(2.0)
    b = a + 3.0
    b.backward()

    assert b.val == 5.0
    assert a.grad == 1.0


def test_mul():
    a = Var(2.0)
    b = Var(3.0)
    c = a * b
    c.backward()

    assert c.val == 6.0
    assert c.grad == 1.0
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_pow():
    a = Var(2.0)
    b = Var(3.0)
    c = a**b
    c.backward()

    assert c.val == 8.0

    assert a.grad == (3.0 * (2.0**2.0))
    assert b.grad == ((2.0**3.0) * math.log(2.0, math.e))


def test_convoluted0():
    # My grad.
    a = Var(2.0)
    b = a + 5.0
    c = (b * 3.0) + a
    d = (c**2.5) * a
    e = (d / a) - c
    f = e + (4 * c)

    f.backward()

    my_a = a
    my_f = f

    # PyTorch.
    a = torch.tensor([2.0], dtype=torch.float64)
    a.requires_grad = True

    b = a + 5.0
    c = (b * 3.0) + a
    d = (c**2.5) * a
    e = (d / a) - c
    f = e + (4 * c)

    f.backward()

    # Check.
    assert math.isclose(my_f.val, f.data.item())
    assert math.isclose(my_a.grad, a.grad.item())


def test_convoluted1():
    # My grad.
    a = Var(4.0)
    b = (a - 5.0) / 7.0
    c = (a + b) * a.relu()
    d = c / a.ln()
    e = (c.relu() + (c + 100.0).relu()) * (c + 15.0).ln()
    f = (e - b) * (d * c)

    f.backward()

    my_a = a
    my_f = f

    # PyTorch.
    a = torch.tensor([4.0], dtype=torch.float64)
    a.requires_grad = True

    b = (a - 5.0) / 7.0
    c = (a + b) * a.relu()
    d = c / torch.log(a)
    e = (c.relu() + (c + 100.0).relu()) * torch.log(c + 15.0)
    f = (e - b) * (d * c)

    f.backward()

    # Check.
    assert math.isclose(my_f.val, f.data.item())
    assert math.isclose(my_a.grad, a.grad.item())
