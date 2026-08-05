# F1 R2 multiscale 128/144/160 candidate (2026-08-05)

## Objective

Extend the current platform-best R2 attention-localization candidate by adding a
smaller 128-pixel crop to the successful 144/160 pair, while continuing to
exclude the harmful 176-pixel crop.

Current platform best before this segment: `68.46237032883407`, produced by
the 144/160 candidate.

## Exact inference command

```bash
cd /home/lux1/noise/reproducibility/aegis_f1
PYTHONPATH=$PWD python3 -m aegis_clip.cli.infer \
  --checkpoint outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/checkpoints/epoch_3.pt \
  --config configs/f1_flat_mlp_lora_selftrain_r2.yaml \
  --output-dir /home/lux1/noise/outputs/delivery/flat_mlp_lora_selftrain_r2_multiscale_128_144_160_fp32_ep3_l040_f050 \
  --tta horizontal_flip --tta-fusion mean_probabilities \
  --tta-temperature 1.5 --tta-view-weight 0.5 --acknowledge-tta-risk \
  --local-view attention_multiscale --local-crop-sizes 128,144,160 \
  --local-top-k 5 --local-weight 0.4 --local-temperature 1.5 \
  --acknowledge-local-view-risk --prior-alignment-strength 0.85 \
  --acknowledge-balanced-test-prior --batch-size 128 \
  --dump-logits outputs/F1_FLAT_MLP_LORA_SELFTRAIN_R2_FP32/seed42/test_multiscale_128_144_160_fused_logits_ep3_l040_f050_t15.pt
```

## Verification

- Inference mode: `attention_multiscale_flip:topk=5:crops=128-144-160:local_weight=0.4:flip_weight=0.5:t=1.5:balanced_prior=0.85`
- Predictions: 24,967
- Corrupt images: 0
- Submission audit: PASS (24,967 rows, 500 classes, expected checkpoint and inference mode)
- ZIP contents: root-level `pred_results.csv` only
- Changes versus current-best 144/160 predictions: 490 (1.96%)
- Changes versus 144/160/176 predictions: 739
- Changes versus crop-160 predictions: 1,038

## Artifact hashes

- Checkpoint SHA-256: `67efab2bf954139b59df074ccf00c0113cbc6ff96163d6e8d66ffbe553b910a4`
- CSV SHA-256: `0382470340e718969d8baa495bc72a466ceaa40e33238fc33c227e56f343bd4d`
- ZIP SHA-256: `beb8e22dc531649d00f35e44bb95dbadf9ecc905e5762d3e48aa6d3b52a03f2d`
- Manifest SHA-256: `f096bd02f93377e194f417cc83e37d94cf12d3207bda964fa95e47ffb219dd3b`

## Status

Candidate generated and audited; platform score pending. This is suitable for
submission, but it is not promoted over the 144/160 platform best until the
platform result is known.
