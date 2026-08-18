# PRNet pseudo-depth pipeline

PRNet is a 3D face position-map regression network. It is not part of
MSU-MFSD. The dataset supplies videos, face boxes, and genuine/attack labels,
but no facial depth targets. CDCN-style pseudo-depth supervision uses a PRNet
face surface for genuine samples and an all-zero map for attacks.

This project includes a modern PyTorch inference implementation. It does not
require a separate PRNet repository or environment.

## Integrated preparation

After `cdcn-msu prepare` has created `cache/frames.jsonl`, run:

```bash
uv run --extra cu130 cdcn-msu prepare-prnet
```

The command performs four stages:

1. Fetches `latest.pth` and `face_ind.txt` from a pinned public revision and
   verifies their exact size and Git blob checksum.
2. Exports genuine 256×256 face crops to `cache/prnet-export/inputs/` and writes
   `cache/prnet-export/contract.jsonl`.
3. Runs batched PRNet inference and rasterizes image-aligned, normalized 32×32
   maps under `cache/prnet-output/`.
4. Validates every result before atomically importing it to
   `cache/depth/<video-id>/<frame-index>.npy`.

The generator skips existing valid maps and replaces corrupt or partial maps.
It writes `cache/prnet-output/generation.json`; the importer writes
`cache/depth/provenance.json`. These hashes are included in local training
provenance and W&B configuration.

Configure resource use in `configs/base.yaml` or with YAML overrides:

```bash
uv run --extra cu130 cdcn-msu prepare-prnet \
  --set prnet.batch_size=8 \
  --set prnet.workers=4 \
  --set prnet.device=cuda
```

On an RTX 3060 12 GB, begin with the default batch size of 16. Reduce
`prnet.batch_size` if CUDA reports an out-of-memory error. Set `prnet.amp=true`
only if the generated maps have been checked for the intended experiment.

## Individual stages

Use these commands to inspect or recover one stage without repeating the rest:

```bash
uv run --extra cu130 cdcn-msu fetch-prnet
uv run --extra cu130 cdcn-msu export-prnet cache/prnet-export
uv run --extra cu130 cdcn-msu generate-prnet
uv run --extra cu130 cdcn-msu import-prnet \
  cache/prnet-export/contract.jsonl cache/prnet-output
```

`generate-prnet --limit 16` is available as an inference smoke test. Run it
again without `--limit` before importing the complete contract.

## Contract format

`cdcn-msu export-prnet <export-root>` writes genuine 256×256 face images under
`<export-root>/inputs/` and a `contract.jsonl`. Every record provides:

- `video_id`
- `frame_index`
- `input_path`
- `output_stem`

Do not change `output_stem`. The built-in generator writes
`<output-stem>.npy`. The importer also accepts compatible results produced by
another implementation. Place each result directly in the supplied output
directory using one supported name:

```text
<output_stem>.npy
<output_stem>.png
<output_stem>.jpg
```

`cdcn-msu import-prnet <contract> <output-root>` rejects missing, non-2D, or
non-finite results. It clips negative values, scales each nonzero map to [0, 1],
bilinearly resizes it to 32×32, and stores a float32 `.npy` map under
`cache/depth/<video-id>/<frame-index>.npy`.

Attack maps are generated natively as zeros and are never exported to PRNet.

## Artifact policy

The artifacts are not committed to this repository. `fetch-prnet` retrieves
them from the `tomguluson92/PRNet_PyTorch` repository at revision
`2d30e244855a043eec071baf78e8edbab9c11c78`. Existing files that fail the pinned
integrity check are rejected rather than silently reused. See
`THIRD_PARTY_NOTICES.md` before redistributing the downloaded files.
