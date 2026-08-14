"""Small deployment check for the Karst Spring Response repository."""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import CFPy
import flopy
import matplotlib
import numpy
import pandas
import streamlit

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "bin" / "CFPv2"

if not EXE.is_file():
    raise FileNotFoundError(f"Missing bundled CFPv2 executable: {EXE}")

if os.name != "nt":
    EXE.chmod(EXE.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if not os.access(EXE, os.X_OK):
        raise PermissionError(f"CFPv2 is not executable: {EXE}")
    with EXE.open("rb") as handle:
        if handle.read(4) != b"\x7fELF":
            raise RuntimeError("bin/CFPv2 is not a Linux ELF executable.")

HASH_FILE = ROOT / "bin" / "CFPv2.sha256"
if HASH_FILE.is_file():
    expected_hash = HASH_FILE.read_text(encoding="utf-8").split()[0]
    digest = hashlib.sha256(EXE.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise RuntimeError("bin/CFPv2 failed its SHA-256 integrity check.")

print("Karst Spring Response deployment smoke test: OK")
print(f"CFPv2:     {EXE}")
print(f"Streamlit: {streamlit.__version__}")
print(f"FloPy:     {flopy.__version__}")
print(f"NumPy:     {numpy.__version__}")
print(f"Pandas:    {pandas.__version__}")
print(f"Matplotlib:{matplotlib.__version__}")
print(f"CFPy:      {Path(CFPy.__file__).resolve()}")
