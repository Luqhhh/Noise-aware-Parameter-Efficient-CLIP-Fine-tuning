# AEGIS F1: noise-aware visual LoRA

## Identity

`AEGIS_F1_VISUAL_LORA_CLEAN_CORE` is not the deprecated legacy `F1-strict` run mentioned in the root README. The legacy run was invalidated by validation leakage. AEGIS F1 was developed in an isolated runner, source commit `d542fc6`, and is preserved under `reproducibility/aegis_f1/`.

## Configuration

- Backbone: official OpenAI CLIP ViT-B/32.
- Parent: Aegis E2 epoch 44, trained only on the official competition training set.
- PEFT: zero-initialised rank-8 LoRA on attention Q, V and output weights in visual blocks 8–11.
- Trainable parameters: 403,956 / 88,253,172 (about 0.46%).
- Noise handling: cross-fitted clean probability threshold 0.70; rejected sample weight 0; GCE q=0.5.
- Regularisation: weak random resized crop + horizontal flip and feature anchoring to the untouched OpenAI CLIP representation.
- Selection: clean-core micro accuracy with a 1% representation-drift budget.
- Best checkpoint: epoch 4; epochs 5–6 did not improve.

## Validation gate

| Model | Raw micro | Clean-core micro | Flip agreement | Feature drift |
|---|---:|---:|---:|---:|
| E2 epoch 44 parent | 70.2307% | 80.7599% | 88.5711% | ~0% |
| AEGIS F1 epoch 4 | 70.6766% | 81.5306% | 88.8426% | 0.4081% |

The gate improved raw accuracy, high-confidence accuracy and flip consistency without exceeding the drift budget. This was used as a safety gate, not as a claim that noisy validation predicts the clean platform test set.

## Platform results

| Inference | Platform | ZIP SHA-256 |
|---|---:|---|
| Bare | **60.5159%** | `6c81b7e38d5688cd67c36cb50868c2de507e0fc4fef3b69b9180c65f29f7a363` |
| Horizontal flip, mean probabilities, T=0.5 | **61.1007%** | `5773f52944af998ac349b7091386282484d8c7dcbc8af296461ae1978dd96657` |

Bare improves the previous registered best bare score (60.2876%) by 0.2283 percentage points. TTA improves the previous registered best TTA score (60.5100%) by 0.5907 points and improves the same F1 checkpoint's bare result by 0.5848 points.

## Compliance

The bare path uses the mandated backbone and official pretrained weights, official-stage training data only, one checkpoint, one linear classifier and one forward pass. No external dataset, class-name enrichment, ensemble or voting is used.

The TTA path uses the same single checkpoint but performs two forward passes (original and horizontal flip) before probability fusion. It is therefore marked as a rules-interpretation risk if “single inference” is interpreted strictly. The bare score is the conservative compliance result.

## Reproduction

Use the isolated runner in `reproducibility/aegis_f1/`; do not substitute the legacy `configs/robust_lora.yaml`. The legacy team LoRA updates only the last block's output projection and is not equivalent to AEGIS F1.

