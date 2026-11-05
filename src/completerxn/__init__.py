"""Data access and scoring for the CompleteRXN benchmark."""

__version__ = "1.0.0"
__all__ = ["DataLoader", "load_split", "load_benchmark", "plot_reaction"]


def __getattr__(name):
    if name == 'DataLoader':
        from .benchmark import DataLoader
        return DataLoader
    raise AttributeError(name)


def load_split(*args, **kwargs):
    """Load a published partition. See completerxn.data.load_split."""
    from .data import load_split as load
    return load(*args, **kwargs)


def load_benchmark(*args, **kwargs):
    """Open the benchmark workflow. See completerxn.benchmark.load_benchmark."""
    from .benchmark import load_benchmark as load
    return load(*args, **kwargs)


def plot_reaction(*args, **kwargs):
    """Draw a completed reaction as a notebook-friendly SVG."""
    from .plotting import plot_reaction as draw
    return draw(*args, **kwargs)
