# External PRNet contract

PRNet is intentionally isolated from this Python 3.12/PyTorch environment. The
adapter exchanges only aligned face images and normalized depth arrays.

## Export

`cdcn-msu export-prnet <export-root>` writes genuine 256×256 face images under
`<export-root>/inputs/` and a `contract.jsonl`. Every record provides:

- `video_id`
- `frame_index`
- `input_path`
- `output_stem`

Run PRNet in its own compatible environment for every exported input. Do not
change `output_stem`.

## Import

Place each result directly in the supplied output directory using one supported
name:

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

