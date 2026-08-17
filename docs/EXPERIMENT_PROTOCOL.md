# Experiment protocol

## Dataset and partitions

MSU-MFSD scene01 contains 280 videos from 35 subjects: 70 genuine videos and
210 attacks split evenly across printed photo, iPhone video, and iPad video.
Android and laptop capture devices each contribute 140 videos.

The official 15-subject training list supplies all development data. Five
deterministic subject-disjoint folds place each development subject in
validation exactly once. The official 20-subject test list remains sealed until
the configuration, epoch count, and validation threshold have been frozen.

## Sampling and targets

- Cache 32 uniformly distributed annotated candidates per video.
- Sample eight candidates per video and training epoch using the run seed.
- Evaluate eight uniformly distributed candidates per video.
- Resize expanded face crops to 256×256; training scale is sampled from
  1.2–1.5 and evaluation scale is fixed at 1.3.
- Normalize RGB values as `(value - 127.5) / 128`.
- Binary supervision uses a 32×32 all-one genuine map and all-zero attack map.
- Pseudo-depth supervision uses normalized external PRNet maps for genuine
  frames and all-zero maps for attacks.

## Optimization

The RTX 3060 default is batch size 8 with two-step gradient accumulation,
yielding effective batch size 16. Training uses FP16 autocast, Adam at `1e-4`,
weight decay `5e-5`, StepLR every 60 epochs with gamma `0.5`, a 200-epoch cap,
and 30-epoch validation-loss patience. `torch.compile` is available but disabled
by default. Seeds are 2026, 2027, and 2028.

The objective is absolute depth MSE plus the eight-neighbor contrastive depth
MSE. Every checkpoint contains model, optimizer, scheduler, scaler, RNG state,
and resolved-configuration hash.

## Comparisons and ablations

The main comparison is CDCN/CDCN++ × binary/pseudo-depth × five folds × three
seeds, totaling 60 CV runs.

One-seed, five-fold ablations vary:

- Central-difference theta: 0.0, 0.3, 0.5, 0.7, and 1.0 using CDCN with
  pseudo-depth supervision.
- Contrastive-loss weight: 0.0, 0.5, and 1.0 using CDCN++ with theta 0.7 and
  pseudo-depth supervision.

## Metrics and final test

The mean predicted depth is the frame score; the mean of eight frame scores is
the video score. Higher scores indicate genuine presentations. Out-of-fold
validation predictions select the threshold minimizing ACER, with accuracy and
then threshold used as deterministic tie-breakers.

The primary metric is ACER. Reports also include APCER, BPCER, ROC-AUC, EER,
accuracy, error counts, attack-type results, and device results. Final models
are refit on all development subjects for the median best fold epoch. Each
model/supervision/seed checkpoint is evaluated on the official test set once
with its frozen out-of-fold threshold.

