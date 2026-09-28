# K05 E04 center-crop submission audit

- Candidate: `K05_E04_CENTER_20260928`
- Validation epoch: 4
- Validation macro accuracy: `0.7264364957809448`
- Validation micro accuracy: `0.7360886931419373`
- Checkpoint SHA-256: `3a1393e0fb464b90e7722cec90e998061b969ab55231c21bdfe5ee4600b32dd6`
- Prediction CSV SHA-256: `b0692d7efb6b87f2be895d2d8c4c23236004298efa887e20062712094e63a3f1`
- Submission ZIP SHA-256: `008be8da3d0787d4b5418ed5baa75aaf565e39b2b85c1939b53e9afce6bd4e83`
- Test predictions: `37,444`
- Corrupt images: `0`
- Inference: global center crop only; no TTA, local crop, prior alignment, or checkpoint blend
- Structural audit: passed (complete test coverage, no duplicates, labels `0000`-`0749`, ZIP contains only root-level `pred_results.csv`)
- Difference from previously submitted `L05_T14_P060`: `6,919 / 37,444` labels changed (`18.478261%`)
- Platform verification: pending submission receipt

This is a genuinely new prediction vector, not a re-upload of the protected L05 package.
