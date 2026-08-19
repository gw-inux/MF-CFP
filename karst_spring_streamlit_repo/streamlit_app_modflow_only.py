"""
Karst Spring Response — MODFLOW continuum alternatives
=====================================================

A Streamlit teaching app derived from the CFP karst spring response model.
The discrete CFP conduit is replaced by standard MODFLOW-2005 approaches:

1. No explicit conduit representation (matrix-only reference case).
2. Highly conductive cells along the conduit alignment.
3. MODFLOW Drain (DRN) cells along the conduit alignment.

The numerical geometry, matrix properties, diffuse recharge, spring boundary,
and transient event timing follow the CFP reference model as closely as possible,
while the refined row around the CFP conduit is intentionally removed.

Source-code overview
--------------------
0. Application configuration and constants
1. Shared utilities and input helpers
2. MODFLOW model design and execution
   2.1 Geometry and stress-period construction
   2.2 MODFLOW package construction
   2.3 Binary output parsing
   2.4 Water-budget post-processing
3. Stored-run state and comparison
4. Plotting and diagnostics
5. Streamlit user interface
   5.1 Model setup and run
   5.2 Current result
   5.3 Head diagnostics
   5.4 Flow diagnostics
   5.5 Water budget
   5.6 Stored-run comparison

The app deliberately does not call st.set_page_config(), so it can later be
embedded into a multipage Streamlit application without conflicting page setup.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import streamlit as st

try:
    import flopy
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile
except Exception as exc:  # pragma: no cover - handled in the UI
    flopy = None
    CellBudgetFile = None
    HeadFile = None
    FLOPY_IMPORT_ERROR = exc
else:
    FLOPY_IMPORT_ERROR = None


# =============================================================================
# 0. APPLICATION CONFIGURATION AND CONSTANTS
# =============================================================================

MODEL_NAME = "Karst_MODFLOW_alternatives"
APP_DIR = Path(__file__).resolve().parent
MAX_STORED_RUNS = 5

# Same stable run colors used conceptually in the CFP app.
RUN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MATRIX_COLOR = "#4c566a"
SECONDARY_COLOR = "#8c564b"

# -----------------------------------------------------------------------------
# Geometry inherited from the CFP reference model, but without row refinement.
# CFP domain: 35 x 50 m = 1750 m in x; approximately 750 m in y.
# Here: 35 columns x 15 rows, all 50 m x 50 m.
# -----------------------------------------------------------------------------
N_LAY = 1
N_ROW = 15
N_COL = 35
DELR = 50.0
DELC = 50.0
TOP = 100.0
BOTTOM = 0.0

DOMAIN_X = N_COL * DELR
DOMAIN_Y = N_ROW * DELC

CONDUIT_ROW = N_ROW // 2  # 0-based, center row
SPRING_COL = 0
SINKHOLE_COL = int(np.floor(1225.0 / DELR))  # same x-location as CFP node 25
CONDUIT_COLS = np.arange(SPRING_COL, SINKHOLE_COL + 1, dtype=int)

SPRING_HEAD = 5.0
INITIAL_HEAD = 5.0

# Matrix reference properties from the CFP app.
DEFAULT_MATRIX_K = 1.0e-5  # m/s
DEFAULT_SY = 0.01

# Diffuse recharge: 316 mm/year, exactly as the CFP reference app.
DIFFUSE_RECHARGE = 316.0 / (1000.0 * 365.25 * 86400.0)  # m/s

# In the CFP app the point recharge is reported as DIRECT RECHARGE = 0.25 m3/s.
# That rate arose from a 0.005 m/s recharge flux on the refined 1 m x 50 m cell.
# The new uniform grid would change the total flux if the same areal rate were
# reused, so the physically relevant total event inflow is preserved explicitly.
POINT_RECHARGE_RATE = 0.25  # m3/s
EVENT_DURATION_HOURS = 2.0
POST_EVENT_HOURS = 10.0
EVENT_TIMESTEP_SECONDS = 60.0
POST_TIMESTEP_SECONDS = 300.0

# Cell-by-cell output unit for LPF, RCH, WEL and DRN packages.
CBC_UNIT = 53

# Drain representation: fixed drain stage at spring level.
DEFAULT_DRAIN_CONDUCTANCE = 1.0e-5  # m2/s

# Session-state schema. Increment only when stored run structure changes.
APP_STATE_SCHEMA_VERSION = 3


# =============================================================================
# 1. SHARED UTILITIES AND INPUT HELPERS
# =============================================================================


def parameter_input(
    label,
    key,
    default,
    min_value,
    max_value,
    *,
    step=None,
    scale="linear",
    use_number_input=False,
    number_format=None,
    log_steps_per_decade=20,
):
    """Parameter input shared with the other educational Streamlit apps.

    It can switch between slider and number input while preserving the current
    physical value. Logarithmic sliders use physical values on a log-spaced
    ``select_slider`` rather than displaying log10(value).
    """

    if scale not in ("linear", "log"):
        raise ValueError("scale must be 'linear' or 'log'.")
    if min_value >= max_value:
        raise ValueError("min_value must be smaller than max_value.")
    if not min_value <= default <= max_value:
        raise ValueError("default must be between min_value and max_value.")
    if scale == "log" and min_value <= 0:
        raise ValueError("Logarithmic parameters must be positive.")

    value_key = f"{key}__value"

    def update_value(widget_key):
        st.session_state[value_key] = float(st.session_state[widget_key])

    if value_key not in st.session_state:
        st.session_state[value_key] = float(default)

    current_value = float(st.session_state[value_key])
    current_value = min(max(current_value, float(min_value)), float(max_value))
    st.session_state[value_key] = current_value

    if use_number_input:
        widget_key = f"_{key}__number"
        st.session_state[widget_key] = current_value

        if step is not None:
            number_step = float(step)
        elif scale == "log":
            exponent = np.floor(np.log10(current_value))
            number_step = 10.0 ** (exponent - 1)
        else:
            number_step = 0.01

        kwargs = {
            "label": label,
            "min_value": float(min_value),
            "max_value": float(max_value),
            "step": float(number_step),
            "key": widget_key,
            "on_change": update_value,
            "args": (widget_key,),
        }
        if number_format is not None:
            kwargs["format"] = number_format
        elif scale == "log":
            kwargs["format"] = "%.2e"
        st.number_input(**kwargs)

    elif scale == "linear":
        widget_key = f"_{key}__slider"
        st.session_state[widget_key] = current_value
        kwargs = {
            "label": label,
            "min_value": float(min_value),
            "max_value": float(max_value),
            "key": widget_key,
            "on_change": update_value,
            "args": (widget_key,),
        }
        if step is not None:
            kwargs["step"] = float(step)
        if number_format is not None:
            kwargs["format"] = number_format
        st.slider(**kwargs)

    else:
        widget_key = f"_{key}__slider"
        decades = np.log10(max_value) - np.log10(min_value)
        n_intervals = max(1, int(round(decades * log_steps_per_decade)))
        options = np.logspace(
            np.log10(min_value), np.log10(max_value), n_intervals + 1
        )

        if not np.any(np.isclose(options, current_value, rtol=1e-12, atol=0.0)):
            options = np.append(options, current_value)
        options = np.unique(np.sort(options)).tolist()

        st.session_state[widget_key] = current_value
        st.select_slider(
            label,
            options=options,
            key=widget_key,
            format_func=lambda x: f"{x:.2e}",
            on_change=update_value,
            args=(widget_key,),
        )

    return float(st.session_state[value_key])


def build_time_discretization(steady_only: bool) -> dict[str, np.ndarray]:
    """Return the MODFLOW stress-period setup and matching output times."""

    if steady_only:
        perlen = np.array([1.0], dtype=float)
        nstp = np.array([1], dtype=int)
        steady = np.array([True], dtype=bool)
    else:
        perlen = np.array(
            [1.0, EVENT_DURATION_HOURS * 3600.0, POST_EVENT_HOURS * 3600.0],
            dtype=float,
        )
        nstp = np.array(
            [
                1,
                int(EVENT_DURATION_HOURS * 3600.0 / EVENT_TIMESTEP_SECONDS),
                int(POST_EVENT_HOURS * 3600.0 / POST_TIMESTEP_SECONDS),
            ],
            dtype=int,
        )
        steady = np.array([True, False, False], dtype=bool)

    times: list[float] = []
    elapsed = 0.0
    for length, steps in zip(perlen, nstp):
        for value in np.linspace(
            elapsed + length / steps, elapsed + length, int(steps)
        ):
            times.append(float(value))
        elapsed += float(length)

    return {
        "perlen": perlen,
        "nstp": nstp,
        "steady": steady,
        "times": np.asarray(times, dtype=float),
    }


def x_centers() -> np.ndarray:
    return (np.arange(N_COL, dtype=float) + 0.5) * DELR


def y_centers() -> np.ndarray:
    return (np.arange(N_ROW, dtype=float) + 0.5) * DELC


def conduit_x() -> np.ndarray:
    return x_centers()[CONDUIT_COLS]


@st.cache_resource(show_spinner=False)
def native_run_semaphore() -> threading.BoundedSemaphore:
    """Limit simultaneous native MODFLOW processes on Community Cloud.

    Every model run already uses an independent temporary workspace.  The
    semaphore additionally prevents a small Community Cloud instance from being
    overloaded when several users press **Run MODFLOW** at the same time.
    """

    return threading.BoundedSemaphore(value=2)


def _prepare_modflow_executable(path: Path) -> Path:
    """Return an absolute executable path and ensure POSIX execute permission.

    Git normally preserves the executable bit, but repository copies and ZIP
    extraction on Windows can remove it.  Streamlit Community Cloud runs on
    Linux, so repairing the user execute bit here makes deployment more robust
    without changing the binary itself.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MODFLOW executable does not exist: {path}")

    if os.name != "nt" and path.suffix.lower() != ".exe":
        if not os.access(path, os.X_OK):
            try:
                path.chmod(path.stat().st_mode | 0o111)
            except OSError as exc:
                raise PermissionError(
                    f"MODFLOW executable exists but could not be made executable: {path}"
                ) from exc
        if not os.access(path, os.X_OK):
            raise PermissionError(f"MODFLOW file is not executable: {path}")

    return path


def find_modflow_executable() -> Path:
    """Find a MODFLOW-2005-compatible executable locally or on Community Cloud.

    Search order
    ------------
    1. Explicit ``MF2005_EXE`` environment variable.
    2. Executables available through the operating-system ``PATH``.
    3. ``bin`` folders next to the app, one level above the app, and in the
       repository working directory used by Streamlit Community Cloud.
    4. The corresponding non-``bin`` directories for backward compatibility.

    The repository layout used by this project is therefore supported directly::

        MF-CFP/
        └── karst_spring_streamlit_repo/
            ├── karst_spring_response_modflow_cloud.py
            └── bin/
                └── mf2005          # Linux executable on Community Cloud

    Standard ``mf2005`` is preferred.  CFPv2 remains an accepted fallback
    because it is MODFLOW-2005 based and this app writes only ordinary MODFLOW
    packages, preserving the behavior of the local version.
    """

    env_value = os.environ.get("MF2005_EXE")
    if env_value:
        path = Path(env_value).expanduser()
        if path.is_file():
            return _prepare_modflow_executable(path)

    if os.name == "nt":
        standard_names = ["mf2005.exe", "mf2005", "MF2005.exe", "MF2005"]
        fallback_names = ["CFPv2.exe", "CFPv2", "cfpv2.exe", "cfpv2"]
    else:
        # Do not select a Windows .exe on Community Cloud/Linux.
        standard_names = ["mf2005", "mf2005dbl", "MF2005", "MF2005DBL"]
        fallback_names = ["CFPv2", "cfpv2"]

    # First respect a system/user PATH installation.
    for name in standard_names + fallback_names:
        resolved = shutil.which(name)
        if resolved:
            return _prepare_modflow_executable(Path(resolved))

    cwd = Path.cwd().resolve()
    app_dir = APP_DIR.resolve()

    # Community Cloud starts Streamlit from the repository root even when the
    # entrypoint is in a subdirectory.  Include both that root and the app-local
    # bin directory shown in the MF-CFP repository structure.
    candidate_dirs = [
        app_dir / "bin",
        app_dir,
        app_dir.parent / "bin",
        app_dir.parent,
        cwd / "bin",
        cwd,
        cwd / "karst_spring_streamlit_repo" / "bin",
        cwd / "karst_spring_streamlit_repo",
    ]

    checked: list[Path] = []
    seen: set[Path] = set()
    for directory in candidate_dirs:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        for name in standard_names + fallback_names:
            candidate = directory / name
            checked.append(candidate)
            if candidate.is_file():
                return _prepare_modflow_executable(candidate)

    checked_text = "\n".join(f"- {path}" for path in checked)
    raise FileNotFoundError(
        "No MODFLOW-2005 executable was found. For Streamlit Community Cloud, "
        "place the Linux executable at 'karst_spring_streamlit_repo/bin/mf2005'. "
        "Locally, mf2005 may also be available on PATH or through MF2005_EXE.\n\n"
        f"Checked locations:\n{checked_text}"
    )


def model_fingerprint(parameters: dict[str, Any]) -> str:
    """Stable fingerprint used to hide stale current results after input changes."""

    parts: list[str] = []
    for key in sorted(parameters):
        value = parameters[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.16g}")
        else:
            parts.append(f"{key}={value}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _integrate_step_rates(times: np.ndarray, rates: np.ndarray) -> float:
    """Integrate piecewise-constant MODFLOW rates over model output intervals."""

    times = np.asarray(times, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if len(times) != len(rates):
        raise ValueError("Time and rate arrays must have equal length.")
    if len(times) == 0:
        return 0.0
    dt = np.diff(np.concatenate(([0.0], times)))
    return float(np.sum(rates * dt))


def _masked_to_float_array(value: Any) -> np.ndarray:
    """Convert ordinary/masked FloPy output arrays to finite float arrays."""

    arr = np.ma.asarray(value)
    return np.asarray(np.ma.filled(arr, 0.0), dtype=float)


# =============================================================================
# 2. MODFLOW MODEL DESIGN AND EXECUTION
# =============================================================================


# -----------------------------------------------------------------------------
# 2.1 Geometry and stress-period construction
# -----------------------------------------------------------------------------

def build_hydraulic_conductivity(
    matrix_k: float,
    representation: str,
    conduit_k: float,
) -> np.ndarray:
    """Build the one-layer K field for the selected conduit representation."""

    hk = np.full((N_ROW, N_COL), float(matrix_k), dtype=np.float32)
    if representation == "High-K cells":
        hk[CONDUIT_ROW, CONDUIT_COLS] = float(conduit_k)
    return hk


def build_drain_stress_data(conductance: float) -> dict[int, list[list[float]]]:
    """Create a linear DRN representation of the implicit conduit.

    The leftmost spring cell is excluded because it is already a constant-head
    boundary cell. Every remaining conduit-alignment cell drains to the common
    spring/base elevation. Summed DRN discharge therefore represents capture by
    an implicit conduit connected to the spring.
    """

    entries = [
        [0, CONDUIT_ROW, int(col), SPRING_HEAD, float(conductance)]
        for col in CONDUIT_COLS
        if int(col) != SPRING_COL
    ]
    return {0: entries}


# -----------------------------------------------------------------------------
# 2.2 MODFLOW package construction and execution
# -----------------------------------------------------------------------------

def run_modflow_model(
    *,
    representation: str,
    matrix_k: float,
    conduit_k: float,
    drain_conductance: float,
    specific_yield: float,
    steady_only: bool,
) -> dict[str, Any]:
    """Build, run and post-process the MODFLOW-2005 alternative model."""

    if flopy is None:
        raise RuntimeError(f"FloPy could not be imported: {FLOPY_IMPORT_ERROR}")

    executable = find_modflow_executable()
    td = build_time_discretization(steady_only)
    perlen = td["perlen"]
    nstp = td["nstp"]
    steady = td["steady"]
    expected_times = td["times"]

    with tempfile.TemporaryDirectory(prefix="karst_mf_") as tmp:
        workspace = Path(tmp)

        mf = flopy.modflow.Modflow(
            MODEL_NAME,
            exe_name=str(executable),
            model_ws=str(workspace),
            version="mf2005",
        )

        flopy.modflow.ModflowDis(
            mf,
            nlay=N_LAY,
            nrow=N_ROW,
            ncol=N_COL,
            nper=len(perlen),
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTTOM,
            perlen=perlen,
            nstp=nstp,
            steady=steady,
            itmuni=1,  # seconds
            lenuni=2,  # metres
        )

        ibound = np.ones((N_LAY, N_ROW, N_COL), dtype=np.int32)
        ibound[:, :, 0] = -1
        strt = np.full((N_LAY, N_ROW, N_COL), INITIAL_HEAD, dtype=np.float32)
        strt[:, :, 0] = SPRING_HEAD
        flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

        hk = build_hydraulic_conductivity(matrix_k, representation, conduit_k)
        flopy.modflow.ModflowLpf(
            mf,
            ipakcb=CBC_UNIT,
            laytyp=1,
            hk=hk,
            sy=float(specific_yield),
        )

        # Diffuse recharge is constant in every stress period.
        rch_array = np.full((N_ROW, N_COL), DIFFUSE_RECHARGE, dtype=np.float32)
        rch_spd = {kper: rch_array for kper in range(len(perlen))}
        flopy.modflow.ModflowRch(
            mf,
            ipakcb=CBC_UNIT,
            nrchop=1,
            rech=rch_spd,
        )

        # The sinkhole event is a true point flux so removing the refined CFP row
        # does not change its total recharge volume.
        if not steady_only:
            wel_spd = {
                0: 0,
                1: [[0, CONDUIT_ROW, SINKHOLE_COL, POINT_RECHARGE_RATE]],
                2: 0,
            }
            flopy.modflow.ModflowWel(
                mf,
                ipakcb=CBC_UNIT,
                stress_period_data=wel_spd,
            )

        if representation == "Drain package":
            flopy.modflow.ModflowDrn(
                mf,
                ipakcb=CBC_UNIT,
                stress_period_data=build_drain_stress_data(drain_conductance),
            )

        flopy.modflow.ModflowPcg(
            mf,
            mxiter=2000,
            iter1=2000,
            npcond=1,
            hclose=1e-3,
            rclose=1e-3,
            relax=0.99,
            nbpol=2,
            iprpcg=5,
            mutpcg=0,
            damp=0.99,
            ihcofadd=9999,
        )

        oc_data: dict[tuple[int, int], list[str]] = {}
        for kper, steps in enumerate(nstp):
            for kstp in range(int(steps)):
                oc_data[(kper, kstp)] = ["save head", "save budget"]
        flopy.modflow.ModflowOc(
            mf,
            stress_period_data=oc_data,
            compact=True,
        )

        mf.write_input()
        success, buffer = mf.run_model(silent=True, report=True)
        if not success:
            tail = "\n".join(buffer[-25:]) if buffer else ""
            raise RuntimeError(
                "The MODFLOW model did not converge."
                + (f"\n\nLast model messages:\n{tail}" if tail else "")
            )

        head_path = workspace / f"{MODEL_NAME}.hds"
        cbc_path = workspace / f"{MODEL_NAME}.cbc"
        if not head_path.exists():
            # FloPy may use .hed depending on executable/version.
            candidates = list(workspace.glob("*.hds")) + list(workspace.glob("*.hed"))
            if not candidates:
                raise FileNotFoundError("MODFLOW head output file was not created.")
            head_path = candidates[0]
        if not cbc_path.exists():
            candidates = list(workspace.glob("*.cbc")) + list(workspace.glob("*.cbb"))
            if not candidates:
                raise FileNotFoundError("MODFLOW cell-budget output file was not created.")
            cbc_path = candidates[0]

        results = read_modflow_outputs(
            head_path=head_path,
            cbc_path=cbc_path,
            expected_times=expected_times,
            representation=representation,
            steady_only=steady_only,
        )

    results.update(
        {
            "representation": representation,
            "matrix_k": float(matrix_k),
            "conduit_k": float(conduit_k),
            "drain_conductance": float(drain_conductance),
            "specific_yield": float(specific_yield),
            "steady_only": bool(steady_only),
            "executable_name": executable.name,
            "hk_field": hk.astype(np.float32, copy=False),
        }
    )
    return results


# -----------------------------------------------------------------------------
# 2.3 Binary output parsing
# -----------------------------------------------------------------------------

def _get_budget_array(
    cbc: CellBudgetFile,
    *,
    totim: float,
    text: str,
) -> np.ndarray:
    """Return one budget term as a full 3-D float array, or zeros if absent."""

    try:
        data = cbc.get_data(totim=float(totim), text=text, full3D=True)
    except Exception:
        return np.zeros((N_LAY, N_ROW, N_COL), dtype=float)

    if data is None or len(data) == 0:
        return np.zeros((N_LAY, N_ROW, N_COL), dtype=float)

    # get_data normally returns a list of records. Sum multiple matching records.
    if isinstance(data, list):
        arrays = [_masked_to_float_array(item) for item in data]
        arr = np.sum(arrays, axis=0) if arrays else np.zeros((N_LAY, N_ROW, N_COL))
    else:
        arr = _masked_to_float_array(data)

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.shape != (N_LAY, N_ROW, N_COL):
        try:
            arr = arr.reshape((N_LAY, N_ROW, N_COL))
        except ValueError as exc:
            raise ValueError(f"Unexpected {text!r} budget shape: {arr.shape}") from exc
    return arr


def _get_flow_right_face(cbc: CellBudgetFile, totim: float) -> np.ndarray:
    return _get_budget_array(cbc, totim=totim, text="FLOW RIGHT FACE")


def read_modflow_outputs(
    *,
    head_path: Path,
    cbc_path: Path,
    expected_times: np.ndarray,
    representation: str,
    steady_only: bool,
) -> dict[str, Any]:
    """Read heads and cell-by-cell flows for all model output times."""

    hds = HeadFile(str(head_path))
    hds_times = np.asarray(hds.get_times(), dtype=float)
    head_arrays = np.asarray(hds.get_alldata(), dtype=np.float32)
    hds.close()

    if head_arrays.ndim == 4:
        head_arrays = head_arrays[:, 0, :, :]
    if head_arrays.ndim != 3:
        raise ValueError(f"Unexpected head-array shape: {head_arrays.shape}")

    # Align requested model times to actual saved head times robustly.
    head_indices: list[int] = []
    for t in expected_times:
        idx = int(np.argmin(np.abs(hds_times - float(t))))
        tolerance = max(1e-5, abs(float(t)) * 1e-6)
        if abs(hds_times[idx] - float(t)) > tolerance:
            raise ValueError(
                f"Could not align requested time {t} s to binary head output."
            )
        head_indices.append(idx)
    heads = head_arrays[np.asarray(head_indices, dtype=int), :, :]

    cbc = CellBudgetFile(str(cbc_path), precision="auto")
    cbc_times = np.asarray(cbc.get_times(), dtype=float)

    spring_cell_flow = np.zeros(len(expected_times), dtype=float)
    boundary_outflow_other = np.zeros(len(expected_times), dtype=float)
    drain_outflow_total = np.zeros(len(expected_times), dtype=float)
    storage_release = np.zeros(len(expected_times), dtype=float)
    diffuse_rate = np.zeros(len(expected_times), dtype=float)
    point_rate = np.zeros(len(expected_times), dtype=float)
    flow_right_face = np.zeros((len(expected_times), N_ROW, N_COL), dtype=np.float32)
    drain_cell_flux = np.zeros((len(expected_times), len(CONDUIT_COLS)), dtype=np.float32)

    for out_idx, t in enumerate(expected_times):
        cbc_idx = int(np.argmin(np.abs(cbc_times - float(t))))
        cbc_t = float(cbc_times[cbc_idx])
        tolerance = max(1e-5, abs(float(t)) * 1e-6)
        if abs(cbc_t - float(t)) > tolerance:
            raise ValueError(
                f"Could not align requested time {t} s to cell-budget output."
            )

        chd = _get_budget_array(cbc, totim=cbc_t, text="CONSTANT HEAD")
        # MODFLOW cell-budget convention: negative is water leaving the cell/model.
        spring_q = float(chd[0, CONDUIT_ROW, SPRING_COL])
        spring_cell_flow[out_idx] = max(-spring_q, 0.0)

        chd_out = np.clip(-chd[0], 0.0, None)
        chd_out[CONDUIT_ROW, SPRING_COL] = 0.0
        boundary_outflow_other[out_idx] = float(np.sum(chd_out))

        storage = _get_budget_array(cbc, totim=cbc_t, text="STORAGE")
        storage_release[out_idx] = float(np.sum(storage))

        recharge = _get_budget_array(cbc, totim=cbc_t, text="RECHARGE")
        diffuse_rate[out_idx] = float(np.sum(np.clip(recharge, 0.0, None)))

        wells = _get_budget_array(cbc, totim=cbc_t, text="WELLS")
        if not np.any(wells):
            wells = _get_budget_array(cbc, totim=cbc_t, text="WELL")
        point_rate[out_idx] = float(np.sum(np.clip(wells, 0.0, None)))

        frf = _get_flow_right_face(cbc, cbc_t)
        flow_right_face[out_idx] = frf[0]

        if representation == "Drain package":
            drains = _get_budget_array(cbc, totim=cbc_t, text="DRAINS")
            if not np.any(drains):
                # Some executables use singular text.
                drains = _get_budget_array(cbc, totim=cbc_t, text="DRAIN")
            drain_outflow_total[out_idx] = float(np.sum(np.clip(-drains[0], 0.0, None)))
            for local_idx, col in enumerate(CONDUIT_COLS):
                drain_cell_flux[out_idx, local_idx] = max(
                    -float(drains[0, CONDUIT_ROW, int(col)]), 0.0
                )

    cbc.close()

    # Flow toward the spring is leftward. FLOW RIGHT FACE is positive to the
    # right, so multiply by -1 along the conduit-alignment row.
    conduit_face_flow = -flow_right_face[:, CONDUIT_ROW, : SINKHOLE_COL]
    face_x = (np.arange(SINKHOLE_COL, dtype=float) + 1.0) * DELR

    # Defensive fallbacks for executables that do not write package cell-budget
    # records despite ipakcb.  The fallback point recharge preserves the exact
    # 0.25 m3/s event, while diffuse recharge excludes the fixed-head column,
    # matching MODFLOW's active recharge budget rather than the geometric domain.
    if not np.any(diffuse_rate):
        recharge_area = (N_COL - 1) * DELR * DOMAIN_Y
        diffuse_rate[:] = DIFFUSE_RECHARGE * recharge_area

    if not steady_only and not np.any(point_rate):
        event_end = 1.0 + EVENT_DURATION_HOURS * 3600.0
        point_rate[(expected_times > 1.0) & (expected_times <= event_end + 1e-9)] = (
            POINT_RECHARGE_RATE
        )

    budget = build_budget_summary(
        times=expected_times,
        steady_only=steady_only,
        diffuse_rate=diffuse_rate,
        point_rate=point_rate,
        storage_release=storage_release,
        spring_cell_flow=spring_cell_flow,
        boundary_outflow_other=boundary_outflow_other,
        drain_outflow_total=drain_outflow_total,
    )

    return {
        "times": np.asarray(expected_times, dtype=float),
        "heads": heads.astype(np.float32, copy=False),
        "spring_flow": spring_cell_flow,
        "flow": spring_cell_flow,  # comparison alias
        "matrix_boundary_outflow": boundary_outflow_other,
        "drain_outflow_total": drain_outflow_total,
        "storage_release_rate": storage_release,
        "diffuse_recharge_rate": diffuse_rate,
        "point_recharge_rate": point_rate,
        "flow_right_face": flow_right_face,
        "conduit_face_flow": conduit_face_flow.astype(np.float32, copy=False),
        "conduit_face_x": face_x,
        "drain_cell_flux": drain_cell_flux,
        "budget": budget,
        "x_centers": x_centers(),
        "y_centers": y_centers(),
        "conduit_columns": CONDUIT_COLS.copy(),
        "conduit_x": conduit_x(),
    }


# -----------------------------------------------------------------------------
# 2.4 Water-budget post-processing
# -----------------------------------------------------------------------------

def build_budget_summary(
    *,
    times: np.ndarray,
    steady_only: bool,
    diffuse_rate: np.ndarray,
    point_rate: np.ndarray,
    storage_release: np.ndarray,
    spring_cell_flow: np.ndarray,
    boundary_outflow_other: np.ndarray,
    drain_outflow_total: np.ndarray,
) -> dict[str, Any]:
    """Return either final steady rates or cumulative transient volumes."""

    if steady_only:
        values = {
            "Diffuse recharge": float(diffuse_rate[-1]),
            "Point recharge": 0.0,
            "Storage release": 0.0,
            "Spring-cell outflow": float(spring_cell_flow[-1]),
            "Other matrix-boundary outflow": float(boundary_outflow_other[-1]),
            "Drain inflow": float(drain_outflow_total[-1]),
        }
        inflow = values["Diffuse recharge"]
        outflow = (
            values["Spring-cell outflow"]
            + values["Other matrix-boundary outflow"]
            + values["Drain inflow"]
        )
        residual = inflow - outflow
        return {
            "mode": "rate",
            "unit": "m³/s",
            "values": values,
            "residual": residual,
        }

    values = {
        "Diffuse recharge": _integrate_step_rates(times, diffuse_rate),
        "Point recharge": _integrate_step_rates(times, point_rate),
        "Storage release": _integrate_step_rates(times, storage_release),
        "Spring-cell outflow": _integrate_step_rates(times, spring_cell_flow),
        "Other matrix-boundary outflow": _integrate_step_rates(
            times, boundary_outflow_other
        ),
        "Drain inflow": _integrate_step_rates(times, drain_outflow_total),
    }
    residual = (
        values["Diffuse recharge"]
        + values["Point recharge"]
        + values["Storage release"]
        - values["Spring-cell outflow"]
        - values["Other matrix-boundary outflow"]
        - values["Drain inflow"]
    )
    return {
        "mode": "cumulative",
        "unit": "m³",
        "values": values,
        "residual": residual,
    }


# =============================================================================
# 3. STORED-RUN STATE AND COMPARISON
# =============================================================================


def initialize_state() -> None:
    if st.session_state.get("mf_alt_schema") != APP_STATE_SCHEMA_VERSION:
        st.session_state.mf_alt_runs = []
        st.session_state.mf_alt_current = None
        st.session_state.mf_alt_count = 0
        st.session_state.mf_alt_schema = APP_STATE_SCHEMA_VERSION

    st.session_state.setdefault("mf_alt_runs", [])
    st.session_state.setdefault("mf_alt_current", None)
    st.session_state.setdefault("mf_alt_count", 0)


def clear_results_for_mode_switch() -> None:
    """Clear completed results when switching steady/transient model mode.

    Steady-state and transient simulations are intentionally kept in separate
    comparison histories because their primary outputs and budget units are not
    directly comparable. The model-input values themselves are preserved.
    """

    st.session_state.mf_alt_runs = []
    st.session_state.mf_alt_current = None
    st.session_state.mf_alt_count = 0

    # Remove result-specific widget state so a fresh history starts cleanly.
    transient_keys = (
        "mf_alt_name_",
        "mf_alt_heads_",
        "mf_alt_flows_",
        "mf_alt_budget_",
        "mf_alt_diag_time_",
        "mf_alt_diag_col_",
    )
    for key in list(st.session_state.keys()):
        if key.startswith(transient_keys):
            del st.session_state[key]

    for key in (
        "mf_alt_compare_ids",
        "mf_alt_compare_budget",
        "mf_alt_steady_compare_col",
    ):
        st.session_state.pop(key, None)


def store_run(result: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    count = int(st.session_state.mf_alt_count) + 1
    slot = ((count - 1) % MAX_STORED_RUNS) + 1
    run = {
        **result,
        "parameters": dict(parameters),
        "fingerprint": model_fingerprint(parameters),
        "execution_number": count,
        "slot_number": slot,
        "name": f"run{slot}",
        "color": RUN_COLORS[slot - 1],
    }

    existing = [
        item for item in st.session_state.mf_alt_runs if int(item["slot_number"]) != slot
    ]
    existing.append(run)
    existing.sort(key=lambda item: int(item["execution_number"]))
    st.session_state.mf_alt_runs = existing[-MAX_STORED_RUNS:]
    st.session_state.mf_alt_current = run
    st.session_state.mf_alt_count = count
    return run


def update_current_name(new_name: str) -> None:
    current = st.session_state.mf_alt_current
    if current is None:
        return
    clean = new_name.strip() or f"run{current['slot_number']}"
    current["name"] = clean
    for item in st.session_state.mf_alt_runs:
        if int(item["execution_number"]) == int(current["execution_number"]):
            item["name"] = clean
            break


def current_if_fresh(parameters: dict[str, Any]) -> dict[str, Any] | None:
    current = st.session_state.mf_alt_current
    if current is None:
        return None
    if current.get("fingerprint") != model_fingerprint(parameters):
        return None
    return current


# =============================================================================
# 4. PLOTTING AND DIAGNOSTICS
# =============================================================================


def _nice_limits(values: np.ndarray, padding: float = 0.05) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if math.isclose(lo, hi):
        delta = max(abs(lo) * 0.05, 0.1)
        return lo - delta, hi + delta
    span = hi - lo
    return lo - padding * span, hi + padding * span


def plot_model_discretization(
    representation: str,
    steady_only: bool,
) -> plt.Figure:
    """Plot the model grid and the currently active conceptual features.

    The preview is purely descriptive; it uses the same constants and conduit
    alignment as the numerical model but does not modify model input.
    """

    # 0 = regular matrix, 1 = specified-head boundary,
    # 2 = high-K conduit cells, 3 = drain cells.
    feature = np.zeros((N_ROW, N_COL), dtype=int)
    if representation == "High-K cells":
        feature[CONDUIT_ROW, CONDUIT_COLS] = 2
    elif representation == "Drain package":
        drain_cols = [int(col) for col in CONDUIT_COLS if int(col) != SPRING_COL]
        feature[CONDUIT_ROW, drain_cols] = 3

    # The specified-head boundary is the complete left column and therefore
    # takes precedence visually over a conduit feature in the spring cell.
    feature[:, SPRING_COL] = 1

    cmap = ListedColormap(["#f7f7f7", "#9ecae1", "#fdae6b", "#a1d99b"])

    fig, ax = plt.subplots(figsize=(11.0, 5.1))
    ax.imshow(
        feature,
        origin="lower",
        interpolation="none",
        extent=(0.0, DOMAIN_X, 0.0, DOMAIN_Y),
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        aspect="equal",
    )

    # Draw the complete numerical grid without overloading the axis labels.
    ax.set_xticks(np.arange(0.0, DOMAIN_X + 0.1, 250.0))
    ax.set_yticks(np.arange(0.0, DOMAIN_Y + 0.1, 250.0))
    ax.set_xticks(np.arange(0.0, DOMAIN_X + DELR, DELR), minor=True)
    ax.set_yticks(np.arange(0.0, DOMAIN_Y + DELC, DELC), minor=True)
    ax.grid(which="minor", color="0.72", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Point recharge is an active model feature only in transient mode.
    if not steady_only:
        sink_x = x_centers()[SINKHOLE_COL]
        sink_y = y_centers()[CONDUIT_ROW]
        ax.scatter(
            [sink_x],
            [sink_y],
            marker="v",
            s=95,
            facecolor=SECONDARY_COLOR,
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
            label="Point recharge (2 h)",
        )

    # Build a compact legend using dummy artists so only active features appear.
    ax.plot([], [], marker="s", ms=10, ls="none", color="#9ecae1", label="Specified head")
    if representation == "High-K cells":
        ax.plot([], [], marker="s", ms=10, ls="none", color="#fdae6b", label="High-K conduit cells")
    elif representation == "Drain package":
        ax.plot([], [], marker="s", ms=10, ls="none", color="#a1d99b", label="DRN cells")

    ax.set_xlim(0.0, DOMAIN_X)
    ax.set_ylim(0.0, DOMAIN_Y)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Model discretization and active features")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_current_response(run: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    t = np.asarray(run["times"], dtype=float)

    # For the DRN representation, the physically comparable response is the
    # water entering the drain system. We assume this captured water is routed
    # rapidly to the spring, but deliberately label it "Drain inflow" rather
    # than "spring outflow". For the other representations, use the actual
    # constant-head spring-cell outflow.
    if run["representation"] == "Drain package":
        q = np.asarray(run["drain_outflow_total"], dtype=float)
        response_label = "Drain inflow"
    else:
        q = np.asarray(run["spring_flow"], dtype=float)
        response_label = "Spring outflow"

    if run["steady_only"]:
        ax.scatter([0.0], [q[-1]], s=70, color=run["color"], label=response_label)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel("Steady state")
    else:
        ax.plot(t / 3600.0, q, lw=2.2, color=run["color"], label=response_label)
        ax.plot(
            t / 3600.0,
            np.asarray(run["point_recharge_rate"], dtype=float),
            lw=1.4,
            ls="--",
            color=SECONDARY_COLOR,
            label="Point recharge",
        )
        ax.set_xlabel("Time [h]")

    ax.set_ylabel("Discharge [m³/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_head_plan(run: dict[str, Any], time_index: int, selected_col: int) -> plt.Figure:
    heads = np.asarray(run["heads"], dtype=float)
    head = heads[int(time_index)]
    x = np.asarray(run["x_centers"], dtype=float)
    y = np.asarray(run["y_centers"], dtype=float)
    xx, yy = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    global_min = float(np.nanmin(heads))
    global_max = float(np.nanmax(heads))
    if math.isclose(global_min, global_max):
        levels = np.linspace(global_min - 0.1, global_max + 0.1, 11)
    else:
        levels = np.linspace(global_min, global_max, 15)

    cf = ax.contourf(xx, yy, head, levels=levels, extend="both")
    cs = ax.contour(xx, yy, head, levels=levels[::2], linewidths=0.7, colors="k", alpha=0.55)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2g")
    fig.colorbar(cf, ax=ax, label="Hydraulic head [m]")

    cx = conduit_x()
    cy = np.full_like(cx, y[CONDUIT_ROW])
    ax.plot(cx, cy, lw=2.3, color=run["color"], label="Conduit alignment")
    ax.scatter([x[SPRING_COL]], [y[CONDUIT_ROW]], s=80, marker="*", color=run["color"], zorder=5, label="Spring cell")
    ax.scatter([x[SINKHOLE_COL]], [y[CONDUIT_ROW]], s=65, marker="v", color=SECONDARY_COLOR, zorder=5, label="Sinkhole cell")
    ax.axvline(x[selected_col], lw=1.5, ls="--", color=MATRIX_COLOR, label="Perpendicular profile")
    ax.scatter([x[selected_col]], [y[CONDUIT_ROW]], s=70, facecolor="white", edgecolor=run["color"], linewidth=2.0, zorder=6)

    ax.set_xlim(0.0, DOMAIN_X)
    ax.set_ylim(0.0, DOMAIN_Y)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_head_profiles(run: dict[str, Any], time_index: int, selected_col: int) -> tuple[plt.Figure, plt.Figure]:
    heads = np.asarray(run["heads"], dtype=float)
    head = heads[int(time_index)]
    x = np.asarray(run["x_centers"], dtype=float)
    y = np.asarray(run["y_centers"], dtype=float)
    hlim = _nice_limits(heads)

    fig1, ax1 = plt.subplots(figsize=(9.5, 4.4))
    ax1.plot(x[CONDUIT_COLS], head[CONDUIT_ROW, CONDUIT_COLS], lw=2.0, color=run["color"])
    ax1.scatter([x[selected_col]], [head[CONDUIT_ROW, selected_col]], s=70, facecolor=run["color"], edgecolor="black", zorder=5)
    ax1.set_xlabel("Distance along conduit alignment [m]")
    ax1.set_ylabel("Hydraulic head [m]")
    ax1.set_ylim(hlim)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(9.5, 4.4))
    ax2.plot(y, head[:, selected_col], lw=2.0, color=MATRIX_COLOR)
    ax2.scatter([y[CONDUIT_ROW]], [head[CONDUIT_ROW, selected_col]], s=70, facecolor=run["color"], edgecolor="black", zorder=5, label="Matrix head")
    if run["representation"] == "Drain package" and int(selected_col) != SPRING_COL:
        ax2.scatter(
            [y[CONDUIT_ROW]],
            [SPRING_HEAD],
            marker="s",
            s=82,
            facecolor="white",
            edgecolor="0.35",
            linewidth=1.0,
            alpha=0.9,
            zorder=4,
            label="Drain elevation",
        )
        ax2.legend()
    ax2.set_xlabel("y [m]")
    ax2.set_ylabel("Hydraulic head [m]")
    ax2.set_ylim(hlim)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    return fig1, fig2


def plot_head_timeseries(run: dict[str, Any], selected_col: int) -> plt.Figure:
    heads = np.asarray(run["heads"], dtype=float)
    series = heads[:, CONDUIT_ROW, selected_col]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    if run["steady_only"]:
        ax.scatter([0.0], [series[-1]], s=70, color=run["color"])
        ax.set_xlabel("Steady state")
    else:
        ax.plot(np.asarray(run["times"]) / 3600.0, series, lw=2.0, color=run["color"])
        ax.set_xlabel("Time [h]")
    ax.set_ylabel("Hydraulic head [m]")
    ax.set_ylim(_nice_limits(heads))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_longitudinal_flow(run: dict[str, Any], time_index: int) -> plt.Figure:
    face_x = np.asarray(run["conduit_face_x"], dtype=float)
    flow = np.asarray(run["conduit_face_flow"], dtype=float)[int(time_index)]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(face_x, flow, lw=2.0, marker="s", ms=4.5, color=run["color"])
    ax.scatter([x_centers()[SPRING_COL]], [run["spring_flow"][int(time_index)]], marker="o", s=65, color=run["color"], zorder=5, label="Spring-cell outflow")
    if not run["steady_only"]:
        ax.scatter([x_centers()[SINKHOLE_COL]], [run["point_recharge_rate"][int(time_index)]], marker="v", s=65, color=SECONDARY_COLOR, zorder=5, label="Point recharge")
    ax.set_xlabel("x along conduit alignment [m]")
    ax.set_ylabel("Flow toward spring [m³/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_drain_capture(run: dict[str, Any], time_index: int) -> plt.Figure:
    flux = np.asarray(run["drain_cell_flux"], dtype=float)[int(time_index)]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(run["conduit_x"], flux, lw=1.7, marker="o", ms=4.5, color=MATRIX_COLOR)
    ax.set_xlabel("x along conduit alignment [m]")
    ax.set_ylabel("Drain inflow [m³/s per cell]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_budget(run: dict[str, Any]) -> plt.Figure:
    budget = run["budget"]
    labels = list(budget["values"].keys())
    values = [float(budget["values"][label]) for label in labels]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    bars = ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(f"Budget component [{budget['unit']}]")
    ax.axhline(0.0, lw=0.8, color="black")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def plot_steady_head_comparison_plan(
    reference_run: dict[str, Any],
    runs: list[dict[str, Any]],
    selected_col: int,
) -> list[plt.Figure]:
    """Return full-width steady-state head-comparison plan-view figures.

    The first figure shows hydraulic-head isolines for the first stored steady
    run. Each subsequently selected run gets its own full-width figure showing
    isolines of ``h_reference - h_run``. Positive values therefore denote lower
    heads (drawdown) relative to the reference run.

    All deviation figures use the same contour levels so their line patterns
    and labelled values can be compared directly.
    """

    ref_head = np.asarray(reference_run["heads"], dtype=float)[-1]
    x = np.asarray(reference_run["x_centers"], dtype=float)
    y = np.asarray(reference_run["y_centers"], dtype=float)
    xx, yy = np.meshgrid(x, y)

    comparison_runs = [
        run
        for run in runs
        if int(run["execution_number"]) != int(reference_run["execution_number"])
    ]

    def decorate_axis(ax: plt.Axes) -> None:
        conduit_y = y[CONDUIT_ROW]
        ax.plot(
            conduit_x(),
            np.full(len(CONDUIT_COLS), conduit_y),
            lw=1.8,
            color="0.35",
        )
        ax.axvline(x[int(selected_col)], lw=1.2, ls="--", color=MATRIX_COLOR)
        ax.set_xlim(0.0, DOMAIN_X)
        ax.set_ylim(0.0, DOMAIN_Y)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

    # Use the same full-width dimensions as the main plan-view head result.
    figures: list[plt.Figure] = []

    ref_min = float(np.nanmin(ref_head))
    ref_max = float(np.nanmax(ref_head))
    if math.isclose(ref_min, ref_max):
        ref_levels = np.linspace(ref_min - 0.1, ref_max + 0.1, 9)
    else:
        ref_levels = np.linspace(ref_min, ref_max, 11)

    fig_ref, ax_ref = plt.subplots(figsize=(10.5, 5.0))
    cs_ref = ax_ref.contour(
        xx,
        yy,
        ref_head,
        levels=ref_levels,
        colors="k",
        linewidths=0.9,
    )
    ax_ref.clabel(cs_ref, inline=True, fontsize=8, fmt="%.2g")
    ax_ref.set_title(f"Reference: {reference_run['name']} — hydraulic-head isolines")
    decorate_axis(ax_ref)
    fig_ref.tight_layout()
    figures.append(fig_ref)

    differences = [
        ref_head - np.asarray(run["heads"], dtype=float)[-1]
        for run in comparison_runs
    ]

    if differences:
        max_abs = max(float(np.nanmax(np.abs(diff))) for diff in differences)
        if not np.isfinite(max_abs) or math.isclose(max_abs, 0.0, abs_tol=1.0e-12):
            max_abs = 0.01
        diff_levels = np.linspace(-max_abs, max_abs, 13)

        for run, diff in zip(comparison_runs, differences):
            fig_diff, ax_diff = plt.subplots(figsize=(10.5, 5.0))

            finite_diff = diff[np.isfinite(diff)]
            if finite_diff.size and not math.isclose(
                float(np.nanmin(finite_diff)),
                float(np.nanmax(finite_diff)),
                abs_tol=1.0e-12,
            ):
                cs_diff = ax_diff.contour(
                    xx,
                    yy,
                    diff,
                    levels=diff_levels,
                    colors=run["color"],
                    linewidths=1.0,
                )
                ax_diff.clabel(cs_diff, inline=True, fontsize=8, fmt="%.3g")

                diff_min = float(np.nanmin(diff))
                diff_max = float(np.nanmax(diff))
                if diff_min < 0.0 < diff_max:
                    ax_diff.contour(
                        xx,
                        yy,
                        diff,
                        levels=[0.0],
                        colors="0.25",
                        linewidths=1.2,
                    )
            else:
                ax_diff.text(
                    0.5,
                    0.5,
                    "No head deviation from reference",
                    transform=ax_diff.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                )

            ax_diff.set_title(
                f"{run['name']} — deviation from {reference_run['name']} "
                "(h_ref - h)"
            )
            decorate_axis(ax_diff)
            fig_diff.tight_layout()
            figures.append(fig_diff)

    return figures

def plot_steady_head_profile_comparison(
    runs: list[dict[str, Any]],
    selected_col: int,
) -> tuple[plt.Figure, plt.Figure]:
    """Overlay steady heads along and perpendicular to the conduit alignment."""

    all_heads = np.concatenate(
        [np.asarray(run["heads"], dtype=float)[-1].ravel() for run in runs]
    )
    hlim = _nice_limits(all_heads)

    fig1, ax1 = plt.subplots(figsize=(9.8, 4.6))
    for run in runs:
        head = np.asarray(run["heads"], dtype=float)[-1]
        x = np.asarray(run["x_centers"], dtype=float)
        ax1.plot(
            x[CONDUIT_COLS],
            head[CONDUIT_ROW, CONDUIT_COLS],
            lw=2.0,
            color=run["color"],
            label=run["name"],
        )
    ax1.set_xlabel("Distance along conduit alignment [m]")
    ax1.set_ylabel("Hydraulic head [m]")
    ax1.set_ylim(hlim)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(9.8, 4.6))
    for run in runs:
        head = np.asarray(run["heads"], dtype=float)[-1]
        y = np.asarray(run["y_centers"], dtype=float)
        ax2.plot(
            y,
            head[:, int(selected_col)],
            lw=2.0,
            color=run["color"],
            label=run["name"],
        )

    if any(run["representation"] == "Drain package" for run in runs) and int(selected_col) != SPRING_COL:
        y = np.asarray(runs[0]["y_centers"], dtype=float)
        ax2.scatter(
            [y[CONDUIT_ROW]],
            [SPRING_HEAD],
            marker="s",
            s=82,
            facecolor="white",
            edgecolor="0.35",
            linewidth=1.0,
            alpha=0.9,
            zorder=4,
            label="Drain elevation",
        )

    ax2.set_xlabel("y [m]")
    ax2.set_ylabel("Hydraulic head [m]")
    ax2.set_ylim(hlim)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    return fig1, fig2


def plot_run_comparison(runs: list[dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    for run in runs:
        if run["representation"] == "Drain package":
            response = np.asarray(run["drain_outflow_total"], dtype=float)
            response_name = "drain inflow"
        else:
            response = np.asarray(run["spring_flow"], dtype=float)
            response_name = "spring outflow"

        if run["steady_only"]:
            ax.scatter(
                [0.0],
                [response[-1]],
                s=65,
                color=run["color"],
                label=f"{run['name']} — {response_name} (steady)",
            )
        else:
            ax.plot(
                np.asarray(run["times"]) / 3600.0,
                response,
                lw=2.0,
                color=run["color"],
                label=f"{run['name']} — {response_name}",
            )
    ax.set_xlabel("Time [h] (steady runs shown at t = 0)")
    ax.set_ylabel("Compared discharge [m³/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# =============================================================================
# 5. STREAMLIT USER INTERFACE
# =============================================================================

initialize_state()

st.title("💧 Karst Spring Response — MODFLOW alternatives")
st.markdown(
    "Explore simplified MODFLOW representations of the same idealized karst "
    "catchment used in the CFP spring-response model. The model can be run "
    "**without a conduit**, with **highly conductive cells**, or with the "
    "**Drain package** as a simplified conduit representation."
)

if FLOPY_IMPORT_ERROR is not None:
    st.error(
        "FloPy is required for this app but could not be imported. Install FloPy "
        f"in the Streamlit environment. Import error: {FLOPY_IMPORT_ERROR}"
    )
    st.stop()


# -----------------------------------------------------------------------------
# 5.1 Model setup and run
# -----------------------------------------------------------------------------
st.header("1. Model setup and run")
st.caption(
    "The model uses a 1750 × 750 m, one-layer, 50 m grid. The left boundary is "
    "fixed at 5 m. Diffuse recharge is 316 mm/year. In transient mode, a "
    "0.25 m³/s point recharge is applied at the sinkhole for two hours."
)

input_a, input_b = st.columns([1, 1])
with input_a:
    use_number_inputs = st.toggle(
        "Use number inputs",
        value=False,
        key="mf_alt_number_mode",
        help="Switch between sliders and direct number input while preserving values.",
    )
with input_b:
    steady_only = st.toggle(
        "Steady state only",
        value=False,
        key="mf_alt_steady_only",
        on_change=clear_results_for_mode_switch,
        help=(
            "Run only the diffuse-recharge steady state; omit the two-hour point-recharge "
            "event. Switching between steady and transient mode clears completed runs "
            "because the results are not directly comparable."
        ),
    )

with st.expander("Matrix and conduit representation", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        representation = st.radio(
            "Conduit representation",
            ["High-K cells", "Drain package", "No conduit"],
            horizontal=False,
            key="mf_alt_representation",
        )
    with c2:
        matrix_k = parameter_input(
            "Matrix hydraulic conductivity [m/s]",
            "mf_alt_matrix_k",
            DEFAULT_MATRIX_K,
            1.0e-8,
            1.0e-2,
            scale="log",
            use_number_input=use_number_inputs,
            number_format="%.2e",
        )
        specific_yield = parameter_input(
            "Matrix specific yield [-]",
            "mf_alt_sy",
            DEFAULT_SY,
            0.001,
            0.50,
            step=0.001,
            use_number_input=use_number_inputs,
            number_format="%.3f",
        )
    with c3:
        # Preserve values for inactive representations so switching approaches does
        # not reset the user's previous settings. Only active controls affect the model.
        conduit_k = float(st.session_state.get("mf_alt_conduit_k__value", 1.0))
        drain_conductance = float(
            st.session_state.get("mf_alt_drain_cond__value", DEFAULT_DRAIN_CONDUCTANCE)
        )

        if representation == "High-K cells":
            conduit_k = parameter_input(
                "Conduit-cell hydraulic conductivity [m/s]",
                "mf_alt_conduit_k",
                1.0,
                1.0e-4,
                1.0e6,
                scale="log",
                use_number_input=use_number_inputs,
                number_format="%.2e",
            )
            st.caption("Only the cells along the conduit alignment receive this K value.")

        elif representation == "Drain package":
            drain_conductance = parameter_input(
                "Drain conductance [m²/s]",
                "mf_alt_drain_cond",
                DEFAULT_DRAIN_CONDUCTANCE,
                1.0e-7,
                1.0e-2,
                scale="log",
                use_number_input=use_number_inputs,
                number_format="%.2e",
            )
            st.caption(
                "The matrix remains homogeneous; the selected conductance is used "
                "only by the DRN cells along the conduit alignment."
            )

        else:
            st.info(
                "No conduit-specific package or high-K zone is active. The model "
                "contains only the homogeneous matrix, specified-head boundary, "
                "diffuse recharge, and (in transient mode) point recharge."
            )

with st.expander("Fixed model geometry and recharge", expanded=False):
    st.markdown(
        f"- Grid: **{N_COL} columns × {N_ROW} rows × 1 layer**, 50 × 50 m cells  \n"
        f"- Domain: **{DOMAIN_X:.0f} × {DOMAIN_Y:.0f} m**  \n"
        f"- Conduit alignment: center row, spring cell column 1 to sinkhole cell column {SINKHOLE_COL + 1}  \n"
        f"- Spring/left boundary head: **{SPRING_HEAD:.1f} m**  \n"
        f"- Diffuse recharge: **316 mm/year** = {DIFFUSE_RECHARGE:.3e} m/s  \n"
        f"- Point recharge: **{POINT_RECHARGE_RATE:.2f} m³/s for {EVENT_DURATION_HOURS:.0f} h** (transient mode only)"
    )

st.markdown("##### Model discretization")
fig_model = plot_model_discretization(representation, steady_only)
st.pyplot(fig_model, use_container_width=True)
plt.close(fig_model)
st.caption(
    "The preview uses the same 50 × 50 m grid and feature locations as the "
    "numerical model. Point recharge is shown only when transient mode is active."
)

parameters = {
    "representation": representation,
    "matrix_k": float(matrix_k),
    "conduit_k": float(conduit_k),
    "drain_conductance": float(drain_conductance),
    "specific_yield": float(specific_yield),
    "steady_only": bool(steady_only),
}

if st.button("▶️ Run MODFLOW", type="primary", use_container_width=True):
    try:
        with st.spinner("Running MODFLOW and reading model outputs..."):
            with native_run_semaphore():
                result = run_modflow_model(**parameters)
            current = store_run(result, parameters)
        st.success(
            f"Model run completed and stored automatically as {current['name']}."
        )
    except Exception as exc:
        st.error(f"MODFLOW run failed: {exc}")

current = current_if_fresh(parameters)


# -----------------------------------------------------------------------------
# 5.2 Current result
# -----------------------------------------------------------------------------
if current is not None:
    st.header("2. Current result")
    name_key = f"mf_alt_name_{current['execution_number']}"
    if name_key not in st.session_state:
        st.session_state[name_key] = current["name"]
    new_name = st.text_input("Run name", key=name_key)
    update_current_name(new_name)
    current = st.session_state.mf_alt_current

    if not current["steady_only"]:
        fig = plot_current_response(current)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    m1, m2, m3 = st.columns(3)
    with m1:
        if current["representation"] == "Drain package":
            st.metric("Drain inflow", f"{current['drain_outflow_total'][-1]:.4g} m³/s")
        else:
            st.metric("Spring outflow", f"{current['spring_flow'][-1]:.4g} m³/s")
    with m2:
        if current["representation"] == "Drain package":
            st.metric("Drain conductance", f"{current['drain_conductance']:.3g} m²/s")
        elif current["representation"] == "High-K cells":
            st.metric("Conduit-cell K", f"{current['conduit_k']:.3g} m/s")
        else:
            st.metric("Conduit representation", "None")
    with m3:
        st.metric("Matrix K", f"{current['matrix_k']:.3g} m/s")

    if current["representation"] == "Drain package":
        st.caption(
            "The response shown for the DRN representation is the total **drain inflow**: "
            "water transferred from the matrix into the drain cells. The conceptual "
            "assumption is that this water is routed rapidly to the spring, but it is "
            "not labelled as spring outflow. The matrix constant-head spring-cell flow "
            "is therefore not used for response comparison."
        )
    else:
        st.caption(
            "The spring response is the constant-head cell-by-cell outflow from the "
            "center-row spring cell on the left boundary."
        )

    diag_a, diag_b, diag_c = st.columns(3)
    with diag_a:
        show_heads = st.toggle(
            "Show head diagnostics",
            value=bool(current["steady_only"]),
            key=f"mf_alt_heads_{current['execution_number']}",
        )
    with diag_b:
        show_flows = st.toggle(
            "Show flow diagnostics",
            value=False,
            key=f"mf_alt_flows_{current['execution_number']}",
        )
    with diag_c:
        show_budget = st.toggle(
            "Show water budget",
            value=False,
            key=f"mf_alt_budget_{current['execution_number']}",
        )

    selected_time_index = len(current["times"]) - 1
    selected_col = SINKHOLE_COL

    if show_heads or show_flows:
        st.markdown("##### Diagnostic controls")
        d1, d2 = st.columns(2)
        with d1:
            if current["steady_only"]:
                st.caption("Steady-state model: one diagnostic time is available.")
                selected_time_index = 0
            else:
                indices = list(range(len(current["times"])))
                selected_time_index = st.select_slider(
                    "Diagnostic time",
                    options=indices,
                    value=min(len(indices) - 1, int(np.argmin(np.abs(np.asarray(current["times"]) - (1.0 + 2.0 * 3600.0))))),
                    format_func=lambda i: f"{current['times'][i] / 3600.0:.3f} h",
                    key=f"mf_alt_diag_time_{current['execution_number']}",
                )
        with d2:
            selected_col = st.slider(
                "Conduit-alignment cell",
                min_value=1,
                max_value=len(CONDUIT_COLS),
                value=len(CONDUIT_COLS),
                step=1,
                key=f"mf_alt_diag_col_{current['execution_number']}",
            ) - 1
            selected_col = int(CONDUIT_COLS[selected_col])
            st.caption(
                f"Selected cell center: x = {current['x_centers'][selected_col]:.0f} m, "
                f"row {CONDUIT_ROW + 1}, column {selected_col + 1}."
            )

    # -------------------------------------------------------------------------
    # 5.3 Head diagnostics
    # -------------------------------------------------------------------------
    if show_heads:
        with st.expander("Head diagnostics 1 — matrix head in plan view", expanded=True):
            fig = plot_head_plan(current, selected_time_index, selected_col)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with st.expander("Head diagnostics 2 — along and perpendicular to the conduit alignment", expanded=True):
            fig1, fig2 = plot_head_profiles(current, selected_time_index, selected_col)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        if not current["steady_only"]:
            with st.expander("Head diagnostics 3 — transient head at selected conduit-alignment cell", expanded=True):
                fig = plot_head_timeseries(current, selected_col)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # -------------------------------------------------------------------------
    # 5.4 Flow diagnostics
    # -------------------------------------------------------------------------
    if show_flows:
        if current["steady_only"] and current["representation"] == "Drain package":
            with st.expander("Flow diagnostics — drain inflow along the conduit alignment", expanded=True):
                fig = plot_drain_capture(current, selected_time_index)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption(
                    "Positive values are matrix water entering the DRN cells. The "
                    "conceptual model assumes this captured water is then routed rapidly "
                    "toward the spring."
                )
        else:
            with st.expander("Flow diagnostics 1 — longitudinal flow toward the spring", expanded=True):
                fig = plot_longitudinal_flow(current, selected_time_index)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                if current["representation"] == "Drain package":
                    st.caption(
                        "In DRN mode this curve is ordinary matrix face flow along the "
                        "conduit alignment; water entering the drain system is shown "
                        "separately below."
                    )

            if current["representation"] == "Drain package":
                with st.expander("Flow diagnostics 2 — drain inflow along the conduit alignment", expanded=True):
                    fig = plot_drain_capture(current, selected_time_index)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5.5 Water budget
    # -------------------------------------------------------------------------
    if show_budget:
        with st.expander("Water budget", expanded=True):
            fig = plot_budget(current)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            budget_note = (
                f"Budget residual = {current['budget']['residual']:.3g} "
                f"{current['budget']['unit']}. In transient mode, storage release "
                "is positive when water is released from aquifer storage and negative "
                "when the aquifer gains storage."
            )
            if current["representation"] == "Drain package":
                budget_note += (
                    " Drain inflow is positive water entering the DRN system; from "
                    "the groundwater-model perspective it is an external outflow."
                )
            st.caption(budget_note)

    with st.expander("Current parameter set", expanded=False):
        table = pd.DataFrame(
            {
                "Parameter": [
                    "Conduit representation",
                    "Matrix K [m/s]",
                    "Conduit-cell K [m/s]",
                    "Drain conductance [m²/s]",
                    "Specific yield [-]",
                    "Steady state only",
                ],
                "Value": [
                    current["representation"],
                    current["matrix_k"],
                    current["conduit_k"] if current["representation"] == "High-K cells" else "not used",
                    current["drain_conductance"] if current["representation"] == "Drain package" else "not used",
                    current["specific_yield"],
                    current["steady_only"],
                ],
            }
        )
        st.dataframe(table, hide_index=True, use_container_width=True)

else:
    if st.session_state.mf_alt_current is not None:
        st.info(
            "The model controls have changed since the most recent execution. "
            "Run MODFLOW again to display a current result; the previous result "
            "remains stored below for comparison."
        )


# -----------------------------------------------------------------------------
# 5.6 Stored-run comparison
# -----------------------------------------------------------------------------
st.header("3. Compare stored runs")
runs = list(st.session_state.mf_alt_runs)

if not runs:
    st.caption("No completed runs are stored yet.")
else:
    option_ids = [int(run["execution_number"]) for run in runs]
    run_by_id = {int(run["execution_number"]): run for run in runs}
    default_ids = option_ids

    selected_ids = st.multiselect(
        "Runs to compare",
        options=option_ids,
        default=default_ids,
        format_func=lambda run_id: f"{run_by_id[run_id]['name']} — {run_by_id[run_id]['representation']}",
        key="mf_alt_compare_ids",
    )
    selected_runs = [run_by_id[run_id] for run_id in selected_ids]

    if selected_runs:
        if selected_runs[0]["steady_only"]:
            reference_run = min(
                runs,
                key=lambda run: int(run["execution_number"]),
            )
            st.caption(
                f"Steady-state reference: **{reference_run['name']}**. The reference "
                "top view is shown as hydraulic-head isolines; each later selected run "
                "is shown below it as full-width h_ref - h isolines, so positive values "
                "indicate lower heads than the reference run."
            )

            compare_col_index = st.slider(
                "Conduit-alignment cell for steady head-profile comparison",
                min_value=1,
                max_value=len(CONDUIT_COLS),
                value=len(CONDUIT_COLS),
                step=1,
                key="mf_alt_steady_compare_col",
            ) - 1
            compare_col = int(CONDUIT_COLS[compare_col_index])

            comparison_plan_figures = plot_steady_head_comparison_plan(
                reference_run,
                selected_runs,
                compare_col,
            )
            for fig in comparison_plan_figures:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown("##### Compare steady head profiles")
            steady_profile_runs = [reference_run] + [
                run
                for run in selected_runs
                if int(run["execution_number"]) != int(reference_run["execution_number"])
            ]
            fig_long, fig_perp = plot_steady_head_profile_comparison(
                steady_profile_runs,
                compare_col,
            )
            st.pyplot(fig_long, use_container_width=True)
            plt.close(fig_long)
            st.pyplot(fig_perp, use_container_width=True)
            plt.close(fig_perp)
        else:
            fig = plot_run_comparison(selected_runs)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        rows = []
        for run in selected_runs:
            rows.append(
                {
                    "Run": run["name"],
                    "Representation": run["representation"],
                    "Steady only": run["steady_only"],
                    "Matrix K [m/s]": run["matrix_k"],
                    "Conduit K [m/s]": run["conduit_k"] if run["representation"] == "High-K cells" else np.nan,
                    "Drain conductance [m²/s]": run["drain_conductance"] if run["representation"] == "Drain package" else np.nan,
                    "Sy [-]": run["specific_yield"],
                    "Response type": "Drain inflow" if run["representation"] == "Drain package" else "Spring outflow",
                    "Final compared response [m³/s]": (
                        run["drain_outflow_total"][-1]
                        if run["representation"] == "Drain package"
                        else run["spring_flow"][-1]
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        compare_budget = st.toggle(
            "Compare water budgets",
            value=False,
            key="mf_alt_compare_budget",
        )
        if compare_budget:
            modes = {run["budget"]["mode"] for run in selected_runs}
            if len(modes) > 1:
                st.warning(
                    "Steady-state budget rates and transient cumulative budgets have "
                    "different units and are therefore not combined in one plot."
                )
            else:
                labels = list(selected_runs[0]["budget"]["values"].keys())
                x = np.arange(len(labels), dtype=float)
                width = 0.8 / max(1, len(selected_runs))
                fig, ax = plt.subplots(figsize=(11.0, 5.0))
                for idx, run in enumerate(selected_runs):
                    values = [run["budget"]["values"][label] for label in labels]
                    offset = (idx - (len(selected_runs) - 1) / 2.0) * width
                    ax.bar(x + offset, values, width=width, color=run["color"], label=run["name"])
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=25, ha="right")
                ax.set_ylabel(f"Budget component [{selected_runs[0]['budget']['unit']}]")
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    if st.button("Clear stored runs"):
        st.session_state.mf_alt_runs = []
        st.session_state.mf_alt_current = None
        st.session_state.mf_alt_count = 0
        st.rerun()

# -----------------------------------------------------------------------------
# Deployment diagnostic (local + Streamlit Community Cloud)
# -----------------------------------------------------------------------------
with st.expander("Deployment diagnostic", expanded=False):
    st.caption(
        "This diagnostic does not run the groundwater model. It only checks "
        "whether the Streamlit process can locate the native MODFLOW executable."
    )
    st.write(f"Platform: `{platform.system()} {platform.machine()}`")
    st.write(f"Application directory: `{APP_DIR}`")
    st.write(f"Working directory: `{Path.cwd().resolve()}`")
    try:
        mf2005_path = find_modflow_executable()
        st.write(f"MODFLOW executable: `{mf2005_path}`")
        if os.name == "nt" or os.access(mf2005_path, os.X_OK):
            st.success("MODFLOW-2005 executable is available to the Streamlit process.")
        else:
            st.error("MODFLOW was found, but the file is not executable.")
    except Exception as exc:
        st.error("MODFLOW-2005 executable is not available.")
        st.exception(exc)

