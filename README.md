# autodiff
Automatic differentiation. Implements reverse-mode autodiff.

## Requirements
`python3`, `pytest` and `pytorch` to test gradients.

## Usage
`var.py` contains the `Var` class, which represents a differentiable float-valued variable. The `Var` class implements fundamental ops (add, mul, pow), derived ops (neg, sub, div), right-hand ops, other math ops (ln), and nn ops (relu).

`test_var.py` contains tests for each op and 2 convoluted examples with gradients and values checked by PyTorch. 

## Sources
* Andrej Karpathy: [micrograd](https://github.com/karpathy/micrograd)
