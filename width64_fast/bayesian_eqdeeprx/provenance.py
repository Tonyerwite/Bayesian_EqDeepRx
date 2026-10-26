"""Fingerprint the standalone training sources and fixed DenoiseNN asset."""

from __future__ import annotations

import hashlib
from pathlib import Path


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted((*root.glob("*.py"), root / "pretrained_denoiser.pt")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


LOADED_SOURCE_FINGERPRINT = source_fingerprint()


def require_source_unchanged() -> str:
    if source_fingerprint() != LOADED_SOURCE_FINGERPRINT:
        raise RuntimeError("source or fixed DenoiseNN asset changed; restart training from the last checkpoint")
    return LOADED_SOURCE_FINGERPRINT
