import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cdcn_msu_mfsd.config import PRNetConfig

_REVISION = "2d30e244855a043eec071baf78e8edbab9c11c78"
_RAW_ROOT = f"https://raw.githubusercontent.com/tomguluson92/PRNet_PyTorch/{_REVISION}"


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    filename: str
    url: str
    size: int
    git_blob_sha1: str


@dataclass(frozen=True)
class ArtifactReport:
    name: str
    path: str
    status: str
    size: int
    sha256: str
    source_revision: str


ARTIFACTS = (
    ArtifactSpec(
        name="checkpoint",
        filename="latest.pth",
        url=f"{_RAW_ROOT}/results/latest.pth",
        size=37_773_804,
        git_blob_sha1="9fd229165bf591f57c87f1258cc48dc649e823b6",
    ),
    ArtifactSpec(
        name="face_indices",
        filename="face_ind.txt",
        url=f"{_RAW_ROOT}/utils/uv_data/face_ind.txt",
        size=1_096_675,
        git_blob_sha1="5b2b9ddb37e1fb1ae4471013c80ca0d57b084dbe",
    ),
)


def artifact_paths(config: PRNetConfig) -> dict[str, Path]:
    return {spec.name: config.model_root / spec.filename for spec in ARTIFACTS}


def _hashes(path: Path, expected_size: int) -> tuple[int, str, str]:
    sha1 = hashlib.sha1(usedforsecurity=False)
    sha1.update(f"blob {expected_size}\0".encode())
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            sha1.update(chunk)
            sha256.update(chunk)
    return size, sha1.hexdigest(), sha256.hexdigest()


def _validate(path: Path, spec: ArtifactSpec) -> str:
    size, git_blob_sha1, sha256 = _hashes(path, spec.size)
    if size != spec.size or git_blob_sha1 != spec.git_blob_sha1:
        raise ValueError(f"PRNet artifact integrity check failed: {path}")
    return sha256


def _download(spec: ArtifactSpec, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".download",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(spec.url, timeout=60) as response:
                shutil.copyfileobj(response, temporary)
        sha256 = _validate(temporary_path, spec)
        os.replace(temporary_path, destination)
        return sha256
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def fetch_prnet_artifacts(config: PRNetConfig, force: bool = False) -> list[ArtifactReport]:
    reports = []
    for spec in ARTIFACTS:
        destination = config.model_root / spec.filename
        if destination.exists() and not force:
            sha256 = _validate(destination, spec)
            status = "ready"
        else:
            sha256 = _download(spec, destination)
            status = "downloaded"
        reports.append(
            ArtifactReport(
                name=spec.name,
                path=str(destination),
                status=status,
                size=spec.size,
                sha256=sha256,
                source_revision=_REVISION,
            )
        )
    return reports


def prnet_artifact_status(config: PRNetConfig) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for spec in ARTIFACTS:
        path = config.model_root / spec.filename
        try:
            sha256 = _validate(path, spec) if path.exists() else None
        except ValueError:
            status[spec.name] = {"ready": False, "path": path, "integrity": "invalid"}
        else:
            status[spec.name] = {
                "ready": sha256 is not None,
                "path": path,
                "integrity": "valid" if sha256 is not None else "missing",
                "sha256": sha256,
            }
    return status


def artifact_reports_json(reports: list[ArtifactReport]) -> list[dict[str, object]]:
    return [asdict(report) for report in reports]
