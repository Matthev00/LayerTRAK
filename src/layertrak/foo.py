import torch


def foo(bar: str) -> bool:
    """Summary line.

    Extended description of function.

    Args:
        bar: Description of input argument.

    Returns:
        Description of return value
    """

    return torch.backends.mps.is_available()


if __name__ == "__main__":  # pragma: no cover
    pass
