from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_ENV_VARS = ("SPS_HOME", "DSPS_CONTINUUM_SSP_FILE", "CUE_DATA_DIR")
DEFAULT_PACKAGES = (
    "composed",
    "numpy",
    "scipy",
    "matplotlib",
    "astropy",
    "jax",
    "jaxlib",
    "numpyro",
    "dsps",
    "fsps",
    "sedpy",
    "pcigale",
    "torch",
    "nflows",
)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA256 digest for one file.

    This is intentionally plain: validation artifacts should be auditable by
    opening the sidecar JSON and checking exactly which file was hashed.
    """

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path, *, max_files: int | None = 4096) -> dict[str, Any]:
    """Return a deterministic digest for all regular files in a directory."""

    path = Path(path)
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if max_files is not None and len(files) > int(max_files):
        raise ValueError(
            f"Refusing to hash {len(files)} files under {path}; "
            f"increase max_files if this directory is the intended validation input."
        )
    digest = hashlib.sha256()
    entries = []
    for file in files:
        relative = file.relative_to(path).as_posix()
        file_hash = sha256_file(file)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        entries.append({"path": relative, "sha256": file_hash, "size_bytes": file.stat().st_size})
    return {"sha256": digest.hexdigest(), "n_files": len(files), "files": entries}


def artifact_provenance(path: str | Path, *, max_files: int | None = 4096) -> dict[str, Any]:
    """Describe a validation input/output file or directory."""

    path = Path(path).expanduser()
    exists = path.exists()
    info: dict[str, Any] = {
        "path": str(path.resolve()) if exists else str(path),
        "exists": bool(exists),
    }
    if not exists:
        return info
    if path.is_file():
        info.update(
            {
                "kind": "file",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    elif path.is_dir():
        directory_hash = sha256_directory(path, max_files=max_files)
        info.update({"kind": "directory", **directory_hash})
    else:
        info["kind"] = "other"
    return info


def collect_run_provenance(
    *,
    paths: Mapping[str, str | Path] | Sequence[str | Path] | None = None,
    env_vars: Sequence[str] = DEFAULT_ENV_VARS,
    package_names: Sequence[str] = DEFAULT_PACKAGES,
    seed: int | None = None,
    command_args: Sequence[str] | Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    max_directory_files: int | None = 4096,
) -> dict[str, Any]:
    """Collect reproducibility metadata for a validation run.

    ``paths`` should include the science-critical external products: SSP files,
    Cue data directories, cached reference spectra, catalogs, and any other
    files whose contents affect the calculation.
    """

    root = _find_repo_root(Path(repo_root).resolve() if repo_root is not None else Path.cwd())
    artifacts = {}
    for label, path in _normalize_paths(paths).items():
        artifacts[label] = artifact_provenance(path, max_files=max_directory_files)

    return {
        "schema": "composed.provenance.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_provenance(root),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "packages": {name: _package_version(name) for name in package_names},
        "environment": {name: os.environ.get(name) for name in env_vars},
        "artifacts": artifacts,
        "seed": None if seed is None else int(seed),
        "command_args": _json_safe(command_args if command_args is not None else sys.argv[1:]),
        "extra": _json_safe(dict(extra or {})),
    }


def provenance_path_for(artifact_path: str | Path) -> Path:
    """Return the JSON sidecar path associated with an artifact."""

    artifact_path = Path(artifact_path)
    if artifact_path.suffix:
        return artifact_path.with_suffix(".provenance.json")
    return artifact_path.with_name(artifact_path.name + ".provenance.json")


def write_provenance(provenance: Mapping[str, Any], path: str | Path) -> Path:
    """Write provenance JSON and return its path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(provenance), indent=2, sort_keys=True) + "\n")
    return path


def read_provenance(path: str | Path) -> dict[str, Any]:
    """Read a provenance JSON file."""

    return json.loads(Path(path).read_text())


def require_provenance(artifact_path: str | Path) -> dict[str, Any]:
    """Load an artifact's provenance sidecar, raising a clear error if missing."""

    provenance_path = provenance_path_for(artifact_path)
    if not provenance_path.exists():
        raise FileNotFoundError(
            f"Missing provenance sidecar for {artifact_path}: expected {provenance_path}. "
            "Regenerate this artifact with CompoSED provenance enabled."
        )
    provenance = read_provenance(provenance_path)
    if provenance.get("schema") != "composed.provenance.v1":
        raise ValueError(f"Unsupported provenance schema in {provenance_path}.")
    return provenance


def save_npz_with_provenance(
    path: str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    provenance_paths: Mapping[str, str | Path] | Sequence[str | Path] | None = None,
    seed: int | None = None,
    command_args: Sequence[str] | Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    **arrays,
) -> tuple[Path, Path]:
    """Save a NumPy archive and a provenance sidecar next to it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    if provenance is None:
        provenance = collect_run_provenance(
            paths=provenance_paths,
            seed=seed,
            command_args=command_args,
            extra=extra,
        )
    provenance_path = write_provenance(provenance, provenance_path_for(path))
    return path, provenance_path


def _normalize_paths(paths: Mapping[str, str | Path] | Sequence[str | Path] | None) -> dict[str, Path]:
    if paths is None:
        return {}
    if isinstance(paths, Mapping):
        return {str(label): Path(path) for label, path in paths.items()}
    return {Path(path).name: Path(path) for path in paths}


def _find_repo_root(start: Path) -> Path:
    completed = _run(["git", "rev-parse", "--show-toplevel"], cwd=start)
    if completed.returncode == 0:
        return Path(completed.stdout.strip()).resolve()
    return start.resolve()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    if commit.returncode != 0:
        return {"root": str(repo_root), "available": False, "error": commit.stderr.strip()}
    status_text = status.stdout if status.returncode == 0 else ""
    return {
        "root": str(repo_root),
        "available": True,
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "dirty": bool(status_text.strip()),
        "status_porcelain": status_text.splitlines(),
    }


def _run(args: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.size > 256:
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
