import hashlib
import io
from pathlib import Path

import pytest

from cdcn_msu_mfsd.config import PRNetConfig
from cdcn_msu_mfsd.prnet_artifacts import ArtifactSpec, fetch_prnet_artifacts


def _spec(payload: bytes) -> ArtifactSpec:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return ArtifactSpec(
        "checkpoint", "latest.pth", "https://example.test/model", len(payload), digest.hexdigest()
    )


def test_fetches_and_reuses_verified_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"checkpoint"
    monkeypatch.setattr("cdcn_msu_mfsd.prnet_artifacts.ARTIFACTS", (_spec(payload),))
    monkeypatch.setattr(
        "cdcn_msu_mfsd.prnet_artifacts.urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(payload),
    )
    config = PRNetConfig(model_root=tmp_path)

    first = fetch_prnet_artifacts(config)
    second = fetch_prnet_artifacts(config)

    assert first[0].status == "downloaded"
    assert second[0].status == "ready"
    assert (tmp_path / "latest.pth").read_bytes() == payload


def test_rejects_corrupt_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"checkpoint"
    monkeypatch.setattr("cdcn_msu_mfsd.prnet_artifacts.ARTIFACTS", (_spec(payload),))
    (tmp_path / "latest.pth").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="integrity check failed"):
        fetch_prnet_artifacts(PRNetConfig(model_root=tmp_path))
