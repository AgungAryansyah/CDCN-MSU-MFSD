# CDCN-MSU-MFSD

A modern PyTorch implementation of CDCN and CDCN++ for face anti-spoofing on
MSU-MFSD.

The original CDCN repository is used as a read-only architecture reference. This
repository contains an independent training, evaluation, and experiment pipeline.

## Development

```bash
uv sync --extra cpu
uv run --extra cpu pytest
```

The remote GPU environment uses the `cu130` extra instead of `cpu`.

