"""Keep older script imports working by forwarding them to completerxn.equivalence."""
from src.completerxn import equivalence as _metric

METRIC_VERSION = _metric.METRIC_VERSION
ReactionMatch = _metric.ReactionMatch
compare_reactions = _metric.compare_reactions
reaction_signature = _metric.reaction_signature


def __getattr__(name):
    return getattr(_metric, name)
