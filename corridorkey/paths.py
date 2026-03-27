"""
Writable locations for CorridorKey assets.

Using a path relative to the process cwd (e.g. checkpoints/CorridorKey) breaks on
macOS .app bundles and other read-only installs: os.makedirs raises PermissionError.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_app_user_data_dir() -> str:
    """Cross-platform user-writable app data directory (created on demand by callers)."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "Sammie-Roto"
    elif os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            local = str(Path.home() / "AppData" / "Local")
        base = Path(local) / "Sammie-Roto"
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        if xdg:
            base = Path(xdg) / "Sammie-Roto"
        else:
            base = Path.home() / ".local" / "share" / "Sammie-Roto"
    return str(base)


def _dir_is_writable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return os.access(path, os.W_OK | os.X_OK)
    except OSError:
        return False


def get_corridorkey_checkpoint_dir() -> str:
    """
    Directory for CorridorKey .pth checkpoint(s).

    Prefer repo-local checkpoints/CorridorKey when it can be created and is writable.
    Otherwise use Application Support (macOS) / XDG / LocalAppData so downloads work
    inside signed .app bundles and when the working directory is not writable.
    """
    legacy = _repo_root() / "checkpoints" / "CorridorKey"
    try:
        legacy.mkdir(parents=True, exist_ok=True)
        if _dir_is_writable(legacy):
            return str(legacy)
    except OSError:
        pass

    user_ckpt = Path(get_app_user_data_dir()) / "checkpoints" / "CorridorKey"
    return str(user_ckpt)
