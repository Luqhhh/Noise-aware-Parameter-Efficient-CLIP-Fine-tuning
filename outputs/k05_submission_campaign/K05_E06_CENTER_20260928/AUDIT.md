# K05 E06 center-crop submission audit

- Candidate: `K05_E06_CENTER_20260928`
- Validation epoch: 6
- Validation macro accuracy: `0.7394055724143982`
- Validation micro accuracy: `0.7499327659606934`
- Predicted classes: `745 / 750`
- Head / medium / tail macro: `0.757280 / 0.766633 / 0.694303`
- Checkpoint SHA-256: `592169c84e6c944e590c2ba093118b0bb2b9dad5038753d56a2a46a7424340e8`
- Binding SHA-256: `eb56c04cad09b8797f5e1544ed62f5a339d214a77380857ac968a73b461bea94`
- Prediction CSV SHA-256: `cd59b04acf801b904c293816f3e5905a026f0c549dfc9548d886eb4997fdb77c`
- Submission ZIP SHA-256: `b4062f2dca4d4fb0b1414df9219fe0946a9450c537693ddaa0518c1e931c6061`
- Test predictions: `37,444`
- Corrupt images: `0`
- Inference: global center crop only; no TTA, local crop, prior alignment, or checkpoint blend
- Structural audit: passed (complete test coverage, no duplicates, labels `0000`-`0749`, ZIP contains only root-level `pred_results.csv`)
- Difference from `K05_E04_CENTER_20260928`: `6,000 / 37,444` labels changed (`16.023929%`)
- Difference from previously submitted `L05_T14_P060`: `6,304 / 37,444` labels changed (`16.835808%`)
- Platform verification: pending submission receipt

E06 is a genuinely distinct prediction vector from both E04 and the previously submitted L05 package.
