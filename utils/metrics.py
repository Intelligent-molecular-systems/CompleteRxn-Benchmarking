"""Keep older imports working with the unchanged v1 metric.

The original code is in completerxn.legacy.equivalence_v1. Benchmark scoring uses completerxn.equivalence.
"""

def __getattr__(name):
    from src.completerxn.legacy import equivalence_v1
    return getattr(equivalence_v1, name)
