"""Benchmark the simple rule-based balancer (an example model, not part of the paper) on the extreme OOD test sets.

Follows "Benchmark on all test sets" in the README, using the model in
models/rule_based_balancer.py. Run from the repository root:

    python examples/run_ruleBased_ood.py
"""
import statistics
import sys
from pathlib import Path

from completerxn import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
from rule_based_balancer import RuleBasedBalancer


def main():
    model = RuleBasedBalancer(workers=8)
    loader = DataLoader()
    scores = []
    for rep in range(2):
        test = loader.test(split="extreme_ood", rep=rep, population="filtered")
        result = test.evaluate(test.predict(model))
        result.save(f"results/local/rule_based/extreme_ood_rep{rep}")
        scores.append(result.summary["equivalence_pct"])
        print(f"extreme_ood rep{rep}: {scores[-1]:.1f} % ({len(test)} reactions)")

    print(f"extreme_ood: {statistics.mean(scores):.1f} ± {statistics.stdev(scores):.1f} %")


if __name__ == "__main__":
    main()
