---
license: mit
pretty_name: CompleteRXN CRB
datasets:
- Intelligent-molecular-systems/completeRXN-benchmark
tags:
- chemistry
- reaction-completion
- pytorch
---

# CompleteRXN CRB

A reaction-completion model for the [CompleteRXN benchmark](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark). It takes an incomplete reaction SMILES and predicts the complete reaction, including the missing molecules.

The model is a [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer) fine-tuned on the training set of the **random split, repetition 0** (seed 42). The same weights give both baselines from the paper, depending on how predictions are decoded:

- **CRB** (Constrained Reaction Balancer): beam search constrained so that predictions are balanced for atoms and charge.
- **RB** (Reaction Balancer): plain beam search without those constraints.

## Files

`random_seed42_rep0/model_step_50000.pt` is the checkpoint after 50,000 fine-tuning steps, the one used in the paper. It contains the model weights, vocabulary, training options and optimizer state, in the OpenNMT format used by the [code repository](https://github.com/Intelligent-molecular-systems/CompleteRxn-Benchmarking).

| | |
|---|---|
| Architecture | Transformer encoder–decoder, 4 layers, hidden size 256, 8 attention heads, feed-forward size 2048 |
| Base model | `STEREO_separated_augm_model_average_20.pt` from the [Molecular Transformer models](https://ibm.ent.box.com/v/MolecularTransformerModels) (Schwaller et al., 2019) |
| Training data | Training set of the CompleteRXN random split, repetition 0 |
| Training | 50,000 steps from the base model on one GPU |

## Use

The model needs the older software environment it was trained with. That environment is provided as an Apptainer container, `models/constrained_reaction_balancer/CRB_env.sif`, in the [reproduction dataset](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-reproduction). Download it to the same path in your checkout of the code repository, then run from the repository root:

```bash
python -m pip install -e .

MODEL_REVISION=$(python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Intelligent-molecular-systems/completeRXN-CRB').sha)")

python examples/run_crb.py \
  --model-repo Intelligent-molecular-systems/completeRXN-CRB \
  --model-file random_seed42_rep0/model_step_50000.pt \
  --model-revision "$MODEL_REVISION" \
  --container models/constrained_reaction_balancer/CRB_env.sif \
  --limit 10 \
  --output-dir results/local/crb_example_10
```

This completes ten test reactions, scores them and saves drawings. Use `--limit 0` for the full test set and `--gpu 0` for GPU inference. In Python, the same model is available as `CRBPredictor.from_pretrained` in `completerxn.predictors`.

This checkpoint fits only the random split, repetition 0. Evaluating it on other splits or repetitions would include reactions it was trained on. To reproduce the paper's results for all splits and repetitions without retraining, rescore the saved predictions in the [reproduction dataset](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-reproduction).

## License

The checkpoint is released under the **MIT License**, like the code repository. Keep the copyright and license notice when you redistribute it.

It is fine-tuned from the pretrained Molecular Transformer weights by Schwaller et al., distributed with the MIT-licensed [Molecular Transformer code](https://github.com/pschwllr/MolecularTransformer) without a separate license for the weights. The model code derives from Molecular Transformer and OpenNMT-py; their MIT and BSD-3-Clause notices are in `models/constrained_reaction_balancer/LICENSE.upstream.md` in the code repository. The training data and its sources are described in the [benchmark dataset card](https://huggingface.co/datasets/Intelligent-molecular-systems/completeRXN-benchmark).

## Citation

If you use this model, please cite:

```bibtex
@article{vogel2026completerxn,
  title={CompleteRXN: Toward Completing Open Chemical Reaction Databases},
  author={Vogel, Gabriel and Noordsij, Minouk and Pidko, Evgeny and Weber, Jana M.},
  journal={Advances in Neural Information Processing Systems},
  volume={39},
  year={2026}
}
```

Please also cite the [Molecular Transformer](https://doi.org/10.1021/acscentsci.9b00576) and the original [Reaction Balancer](https://www.sciencedirect.com/science/chapter/bookseries/abs/pii/B978044328824150524X).
