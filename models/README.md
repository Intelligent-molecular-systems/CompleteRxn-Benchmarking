# Models

The benchmark compares three methods:

| Model | Type | Folder |
|---|---|---|
| **CRB** (Constrained Reaction Balancer) | Transformer model with balance constraints | `constrained_reaction_balancer/` |
| **RB** (Reaction Balancer) | The same model without balance constraints | `constrained_reaction_balancer/` |
| **SynRBL** | Rule-based completion using shared molecular structure | `SynRBL/` |

`rule_based_balancer.py` is a small example model that completes reactions by balancing atoms and charge. It is not part of the paper and only illustrates the `predict(inputs)` interface.

RB and CRB share the same fine-tuned weights and training code. CRB applies atom and charge balance constraints when generating predictions. RB does not.

See the README in each subfolder for environment setup, training, inference, and evaluation instructions.
