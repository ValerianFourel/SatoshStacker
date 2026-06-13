"""A sandboxed scratch workspace the analyst LLM may write to.

The operator allowed the bot to create/edit files, but ONLY inside one dedicated
scratch directory — never the repo, never the trading agent, never `.env`. Every
path is resolved and checked to stay within the root, so a traversal like
``../../agent/loop.py`` or an absolute path is rejected. There is no delete and
no execute here — this is a notes/metrics scratchpad, not a shell.
"""
from __future__ import annotations

import os
from pathlib import Path

_MAX_BYTES = 256 * 1024  # cap a single scratch file


class ScratchError(Exception):
    pass


class Scratch:
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, name: str) -> Path:
        if not name or name.strip() in (".", ".."):
            raise ScratchError("empty or invalid filename")
        p = Path(name)
        if p.is_absolute() or p.drive or ".." in p.parts:
            raise ScratchError(f"path escapes scratch sandbox: {name!r}")
        target = (self.root / p).resolve()
        # belt-and-suspenders: confirm the resolved path is under the root
        if self.root != target and self.root not in target.parents:
            raise ScratchError(f"path escapes scratch sandbox: {name!r}")
        return target

    def write(self, name: str, content: str) -> str:
        target = self._resolve(name)
        data = content if isinstance(content, str) else str(content)
        if len(data.encode("utf-8")) > _MAX_BYTES:
            raise ScratchError(f"scratch file too large (> {_MAX_BYTES} bytes)")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data, encoding="utf-8")
        return str(target.relative_to(self.root))

    def read(self, name: str) -> str:
        return self._resolve(name).read_text(encoding="utf-8")

    def list(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")
                      if p.is_file())
