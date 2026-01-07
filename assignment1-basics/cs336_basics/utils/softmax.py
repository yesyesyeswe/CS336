from torch import Tensor, max, sum, exp


def softmax(matrix: Tensor, i: int = -1) -> Tensor:
    max_val, _ = max(matrix, i, keepdim=True)
    exp_vals = exp(matrix - max_val)
    exp_sum = sum(exp_vals, i, keepdim=True)

    return exp_vals / exp_sum
