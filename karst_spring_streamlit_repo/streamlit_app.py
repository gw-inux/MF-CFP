from __future__ import annotations

import os
import re
import stat
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import streamlit as st

import CFPy as cfpy
import flopy
import flopy.utils.binaryfile as bf


# =============================================================================
# SOURCE-CODE STRUCTURE / TABLE OF CONTENTS
# =============================================================================
# Line numbers below refer to this v16 file. Regenerate this overview after
# structural edits that add, remove, or move source-code blocks.
#
# 0. Application configuration .................................. line 70
# 1. Model ...................................................... line 121
#   1.1 Runtime environment and model files ..................... line 124
#   1.2 Time discretization helper .............................. line 241
#   1.3 MODFLOW + CFP model design and execution ................ line 255
#     1.3.1 Initialize MODFLOW/CFP .............................. line 282
#     1.3.2 Continuum characteristics ........................... line 291
#     1.3.3 Time discretization ................................. line 319
#     1.3.4 Boundary and initial conditions ..................... line 338
#     1.3.5 MODFLOW packages .................................... line 352
#     1.3.6 CFP solver variables ................................ line 419
#     1.3.7 CFP conduit-network construction .................... line 428
#     1.3.8 CFP pipe data ....................................... line 456
#     1.3.9 CFP node and exchange data .......................... line 476
#     1.3.10 CFP package and input files ........................ line 490
#     1.3.11 Execute CFP/MODFLOW ................................ line 550
#     1.3.12 External conduit boundary fluxes ................... line 561
#     1.3.13 Cumulative whole-run water budget .................. line 623
#     1.3.14 Head and flow diagnostics .......................... line 637
# 2. Model output and post-processing ........................... line 678
#   2.1 Cumulative water-budget parsing ......................... line 681
#   2.2 CFP listing-file parsing ................................ line 809
#   2.3 MODFLOW matrix-head output and diagnostic assembly ...... line 1157
# 3. User input, run state, and diagnostic selection ............ line 1284
#   3.1 Synchronized numerical-input helpers .................... line 1287
#   3.2 Stored-run data and rolling history ..................... line 1574
#   3.3 Diagnostic node/tube selection and geometry ............. line 1686
# 4. Plotting and diagnostic visualization ...................... line 1782
#   4.1 Common plotting, scale, and formatting helpers .......... line 1785
#   4.2 Spring-response comparison .............................. line 2025
#   4.3 Head diagnostics ........................................ line 2069
#   4.4 Flow diagnostics ........................................ line 2575
#   4.5 Cumulative water-budget plots ........................... line 3114
# 5. Streamlit user interface ................................... line 3214
#   5.1 Session-state initialization and migration .............. line 3217
#   5.2 Model setup, parameter inputs, and model execution ...... line 3269
#   5.3 Current result and optional diagnostics ................. line 3618
#   5.4 Stored-run comparison ................................... line 4311


# =============================================================================
# 0. APPLICATION CONFIGURATION
# =============================================================================
# Page configuration is intentionally not set here. This keeps the file easy to
# integrate later as a page in a larger multipage Streamlit application. Put
# st.set_page_config(...) in the main entry point of that application if needed.

MODEL_NAME = "CFPy_example"
APP_DIR = Path(__file__).resolve().parent
BIN_DIR = APP_DIR / "bin"
BUNDLED_CFP_EXECUTABLE = BIN_DIR / "CFPv2"
MAX_STORED_RUNS = 5
APP_STATE_SCHEMA_VERSION = 7

# One stable color is assigned to each rolling run slot. The same color follows
# that run from the spring-response plot into all conduit-head diagnostics.
# Matrix heads use one separate, constant color in every diagnostic plot.
RUN_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
]
MATRIX_HEAD_COLOR = "#111111"
REFERENCE_COLOR = "#555555"
COMPARISON_REFERENCE_COLOR = "#a65628"
HEAD_COLORMAP = "viridis"

# CFP output marker used in the listing file.
FLOW_RESULTS_MARKER = "RESULTS OF FLOW CALCULATION"

# Generic Fortran/decimal number pattern. D exponents are converted to E later.
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_TUBE_RESULT_RE = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\S+\s+({_NUMBER})(?:\s|$)",
    re.IGNORECASE,
)
_NODE_COORD_RE = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    rf"({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s*$",
    re.IGNORECASE,
)
_STRESS_TIME_RE = re.compile(
    r"STRESS PERIOD/TIME STEP\s+(\d+)\s+(\d+)",
    re.IGNORECASE,
)

MODEL_BUDGET_MARKER = "VOLUMETRIC BUDGET FOR ENTIRE MODEL"


# =============================================================================
# 1. MODEL
# =============================================================================
# -----------------------------------------------------------------------------
# 1.1 Runtime environment and model files
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_run_lock() -> threading.Lock:
    """Return one process-wide lock shared by Streamlit sessions.

    Some CFPy routines read/write files in the process working directory. Since
    os.chdir() is process-wide, two simultaneous model runs could otherwise
    interfere with one another even if each run has its own temporary folder.
    """
    return threading.Lock()


def _ensure_executable_permissions(path: Path) -> Path:
    """Ensure a bundled Linux solver can be executed on Streamlit Cloud.

    Git preserves executable bits when committed from Linux, but repositories
    are often prepared or uploaded from Windows.  Applying the execute bits at
    runtime makes deployment robust in either case and does not modify the
    numerical executable itself.
    """
    path = path.resolve()
    if os.name != "nt":
        try:
            path.chmod(
                path.stat().st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH
            )
        except OSError as exc:
            raise PermissionError(
                f"CFP executable permissions could not be set for {path}: {exc}"
            ) from exc

        if not os.access(path, os.X_OK):
            raise PermissionError(f"CFP executable is not executable: {path}")

    return path


def find_cfp_executable() -> Path:
    """Find CFPv2 locally, preferring the Streamlit repository ``bin/`` copy.

    ``CFP_EXECUTABLE`` remains available as an override for local development.
    The bundled Linux executable is the default on Streamlit Community Cloud; a
    local Windows ``CFPv2.exe`` is retained as a convenience fallback.
    """
    env_path = os.environ.get("CFP_EXECUTABLE")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists() and path.is_file():
            return _ensure_executable_permissions(path)
        raise FileNotFoundError(
            f"CFP_EXECUTABLE points to a file that does not exist: {path}"
        )

    candidates = [
        BUNDLED_CFP_EXECUTABLE,
        BIN_DIR / "cfpv2",
        APP_DIR / "CFPv2",
        APP_DIR / "cfpv2",
        APP_DIR.parent / "CFPv2",
        Path.cwd() / "CFPv2",
        # Local Windows-development fallbacks:
        APP_DIR / "CFPv2.exe",
        APP_DIR.parent / "CFPv2.exe",
        Path.cwd() / "CFPv2.exe",
    ]

    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.exists() and candidate.is_file():
            return _ensure_executable_permissions(candidate)

    checked_text = "\n".join(f"- {candidate}" for candidate in checked)
    raise FileNotFoundError(
        "CFPv2 executable not found. For Streamlit deployment, place the Linux "
        "binary at 'bin/CFPv2'. For local development you may instead set the "
        "CFP_EXECUTABLE environment variable.\n\n"
        f"Checked:\n{checked_text}"
    )


@contextmanager
def working_directory(path: Path):
    """Temporarily change cwd and always restore it afterwards."""
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def find_output_file(workspace: Path, preferred_name: str, suffixes: tuple[str, ...]) -> Path:
    """Find a model output file robustly, including case variations."""
    preferred = workspace / preferred_name
    if preferred.exists():
        return preferred

    suffixes_lower = tuple(s.lower() for s in suffixes)
    for candidate in workspace.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in suffixes_lower:
            if candidate.stem.lower() == MODEL_NAME.lower():
                return candidate

    raise FileNotFoundError(
        f"Expected output file '{preferred_name}' was not created in {workspace}."
    )


# -----------------------------------------------------------------------------
# 1.2 Time discretization helper
# -----------------------------------------------------------------------------
def build_time_vector(perlen: np.ndarray, n_stps: np.ndarray) -> np.ndarray:
    """Return cumulative simulation times at every saved time step."""
    intervals = np.concatenate(
        [
            np.repeat(float(per) / int(nstep), int(nstep))
            for per, nstep in zip(perlen, n_stps)
        ]
    )
    return np.cumsum(intervals)


# -----------------------------------------------------------------------------
# 1.3 MODFLOW + CFP model design and execution
# -----------------------------------------------------------------------------
def cfpy_model(
    dmt: float,
    trtst: float,
    rh: float,
    lcrey: int,
    hcrey: int,
    kxch: float,
    cfptemp: float,
    hk: float,
    sy: float,
    CADS: float,
    laminar_only: bool,
) -> dict:
    """Run the CFPy/MODFLOW model and return discharge plus head/flow diagnostics."""
    executable = find_cfp_executable()

    # This lock is necessary because parts of CFPy use the current working
    # directory. It prevents simultaneous Streamlit sessions from changing cwd
    # underneath one another.
    with get_model_run_lock():
        with tempfile.TemporaryDirectory(prefix="cfpy_streamlit_") as tmp:
            workspace = Path(tmp)

            with working_directory(workspace):
                # -----------------------------------------------------------------
                # 1.3.1 Initialize MODFLOW/CFP
                # -----------------------------------------------------------------
                mf = flopy.modflow.Modflow(
                    MODEL_NAME,
                    exe_name=str(executable),
                    model_ws=str(workspace),
                )

                # -----------------------------------------------------------------
                # 1.3.2 Continuum characteristics
                # -----------------------------------------------------------------
                delr = 50.0
                finer_mesh = np.array([1.5, 2.5, 5.0, 10.0, 20.0, 35.0])
                delc = np.hstack(
                    [
                        np.repeat(50.0, 6),
                        np.flip(finer_mesh),
                        np.array([1.0]),
                        finer_mesh,
                        np.repeat(50.0, 6),
                    ]
                )

                n_rows = len(delc)
                n_cols = 35
                n_lays = 1
                lay_elevs = [100.0, 0.0]

                col_node_1 = int(np.floor(1225.0 / delr))
                row_node_1 = int(np.floor(len(delc) / 2))

                lay_elevs_array = [
                    np.ones((n_rows, n_cols)) * lay_elevs[0],
                    np.ones((n_rows, n_cols)) * lay_elevs[1],
                ]

                # -----------------------------------------------------------------
                # 1.3.3 Time discretization
                # -----------------------------------------------------------------
                time_unit = 1  # seconds
                n_pers = 3

                LP1 = 2  # hours of injection
                LP2 = 10  # hours after injection
                TS1 = 60  # seconds
                TS2 = 300  # seconds

                perlen = np.array([1.0, LP1 * 3600.0, LP2 * 3600.0])
                n_stps = np.array(
                    [1, int(LP1 * 3600 / TS1), int(LP2 * 3600 / TS2)],
                    dtype=int,
                )
                steady = np.array([True, False, False])
                times = build_time_vector(perlen, n_stps)

                # -----------------------------------------------------------------
                # 1.3.4 Boundary and initial conditions
                # -----------------------------------------------------------------
                chb_left = 5.0
                rch_background = (
                    np.ones((n_rows, n_cols))
                    * 316.0
                    / (1000.0 * 365.25 * 86400.0)
                )
                rch_injection = rch_background.copy()
                rch_injection[row_node_1, col_node_1] = 5.0 / 1000.0

                h_init = 5.0

                # -----------------------------------------------------------------
                # 1.3.5 MODFLOW packages
                # -----------------------------------------------------------------
                flopy.modflow.ModflowDis(
                    mf,
                    n_lays,
                    n_rows,
                    n_cols,
                    n_pers,
                    delr,
                    delc,
                    top=lay_elevs[0],
                    botm=lay_elevs[1],
                    perlen=perlen,
                    nstp=n_stps,
                    steady=steady,
                    itmuni=time_unit,
                    lenuni=2,
                )

                ibound = np.ones((n_lays, n_rows, n_cols), dtype=np.int32)
                ibound[:, :, 0] = -1
                h_init_array = (
                    np.ones((n_lays, n_rows, n_cols), dtype=np.float32) * h_init
                )
                h_init_array[:, :, 0] = chb_left
                flopy.modflow.ModflowBas(mf, ibound=ibound, strt=h_init_array)

                flopy.modflow.ModflowLpf(mf, laytyp=1, hk=hk, sy=sy)

                # Save matrix heads at every model time. The original notebook
                # used the default OC package, which saved the binary matrix
                # head only at the initial steady-state step. Full head output
                # is required for the selectable perpendicular transect.
                oc_stress_period_data = {
                    (per_idx, step_idx): ["save head"]
                    for per_idx, nstep in enumerate(n_stps)
                    for step_idx in range(int(nstep))
                }
                flopy.modflow.ModflowOc(
                    mf,
                    stress_period_data=oc_stress_period_data,
                    compact=True,
                )

                flopy.modflow.ModflowPcg(
                    mf,
                    mxiter=2000,
                    iter1=2000,
                    npcond=1,
                    hclose=1e-2,
                    rclose=1e-2,
                    relax=0.99,
                    nbpol=2,
                    iprpcg=5,
                    mutpcg=0,
                    damp=0.99,
                    ihcofadd=9999,
                )

                rech = {
                    0: rch_background,
                    1: rch_injection,
                    2: rch_background,
                }
                flopy.modflow.mfrch.ModflowRch(mf, nrchop=1, rech=rech)

                # -----------------------------------------------------------------
                # 1.3.6 CFP solver variables
                # -----------------------------------------------------------------
                cfptol = 1e-9
                cfprelax = 0.99
                chd_outlet = 5.0
                elev_nodes = 1.0
                cad = CADS

                # -----------------------------------------------------------------
                # 1.3.7 CFP conduit-network construction
                # -----------------------------------------------------------------
                network = np.zeros((n_rows, n_cols))
                network[row_node_1, : (col_node_1 + 1)] = 1.0
                elevations = np.ones((n_rows, n_cols)) * elev_nodes

                validator = cfpy.preprocessing.GeneralValidator(
                    network=network,
                    elevations=elevations,
                )
                validator.validate_network()
                validator.export_network()
                validator.generate_nbr(
                    path=str(workspace) + os.sep,
                    nrows=n_rows,
                    ncols=n_cols,
                    nlays=n_lays,
                    nplanes=1,
                    layer_elevations=lay_elevs_array,
                )

                nbr = cfpy.nbr()
                bot_elev, cond_elev = nbr.nbr_read()
                nbr_data = nbr.nbr(bot_elev, cond_elev)

                mf.write_input()

                # -----------------------------------------------------------------
                # 1.3.8 CFP pipe data
                # -----------------------------------------------------------------
                # ``Laminar only`` leaves the familiar user sliders unchanged but
                # moves the CFP transition thresholds far above the simulated range.
                # This is intentionally done only at model-write time so stored user
                # settings remain transparent.
                lcrey_model = int(lcrey * 10000) if laminar_only else int(lcrey)
                hcrey_model = int(hcrey * 10000) if laminar_only else int(hcrey)

                n_pipes = len(nbr_data[5])
                pipe_data = [
                    nbr_data[5],
                    (np.ones(n_pipes) * dmt).tolist(),
                    (np.ones(n_pipes) * trtst).tolist(),
                    (np.ones(n_pipes) * rh).tolist(),
                    (np.ones(n_pipes) * lcrey_model).tolist(),
                    (np.ones(n_pipes) * hcrey_model).tolist(),
                ]

                # -----------------------------------------------------------------
                # 1.3.9 CFP node and exchange data
                # -----------------------------------------------------------------
                n_head = (np.ones(len(nbr_data[0])) * -1).tolist()
                outlet_pos = nbr_data[2].index([1, row_node_1 + 1, 1])
                n_head[outlet_pos] = chd_outlet
                node_data = [nbr_data[0], n_head]

                kex_data = [
                    nbr_data[0],
                    np.ones(len(nbr_data[0])) * kxch,
                ]
                cads_data = (np.ones(len(nbr_data[0])) * cad).tolist()

                # -----------------------------------------------------------------
                # 1.3.10 CFP package and input files
                # -----------------------------------------------------------------
                cfp_data = cfpy.cfp(
                    mode=1,
                    nnodes=len(nbr_data[0]),
                    npipes=len(nbr_data[5]),
                    nlay=n_lays,
                    nbr_data=nbr_data,
                    geoheight=cond_elev,
                    sa_exchange=1,
                    epsilon=cfptol,
                    niter=2000,
                    relax=cfprelax,
                    p_nr=0,
                    cond_data=pipe_data,
                    n_head=node_data,
                    k_exchange=kex_data,
                    ncl=0,
                    cl=0,
                    ltemp=cfptemp,
                    condl_data=0,
                    cads=cads_data,
                ).cfp()

                coc_data = cfpy.coc(
                    nnodes=len(nbr_data[0]),
                    node_numbers=nbr_data[0],
                    n_nts=1,
                    npipes=len(nbr_data[5]),
                    pipe_numbers=nbr_data[5],
                    t_nts=1,
                ).coc()

                p_crch = np.zeros(len(nbr_data[0])).tolist()
                injection_pos = nbr_data[2].index(
                    [col_node_1 + 1, row_node_1 + 1, 1]
                )
                p_crch[injection_pos] = 1.0
                crch_data = cfpy.crch(
                    iflag_crch=1,
                    nper=n_pers,
                    node_numbers=nbr_data[0],
                    p_crch=p_crch,
                ).crch()

                cfpy.write_input(
                    modelname=MODEL_NAME,
                    data_strings=[coc_data, crch_data, cfp_data],
                    file_extensions=["coc", "crch", "cfp"],
                ).write_input()

                cfpy.update_nam(
                    modelname=MODEL_NAME,
                    mode=1,
                    cfp_unit_num=52,
                    crch_unit_num=53,
                    coc_unit_num=54,
                ).update_nam()

                # -----------------------------------------------------------------
                # 1.3.11 Execute CFP/MODFLOW
                # -----------------------------------------------------------------
                success, model_buffer = mf.run_model(silent=True, report=True)
                if not success:
                    tail = "\n".join(model_buffer[-20:]) if model_buffer else ""
                    raise RuntimeError(
                        "The CFP/MODFLOW model did not converge."
                        + (f"\n\nLast model messages:\n{tail}" if tail else "")
                    )

                # -----------------------------------------------------------------
                # 1.3.12 External conduit boundary fluxes from the CFP node table
                # -----------------------------------------------------------------
                # DIRECT RECHARGE is the imposed sinkhole/direct-conduit inflow.
                # QFIX is the fixed-head-node flux; negative QFIX is outflow to the
                # spring. These are external boundary fluxes and can differ from the
                # adjacent tube Q because water may exchange with the matrix at the
                # boundary node itself.
                listing_file = find_output_file(
                    workspace,
                    f"{MODEL_NAME}.list",
                    (".list", ".lst"),
                )
                listing_results = parse_cfp_listing(listing_file, times)

                spring_flow = np.asarray(
                    listing_results["spring_outflow"], dtype=float
                )
                inlet_flow = np.asarray(
                    listing_results["direct_recharge_total"], dtype=float
                )

                if len(spring_flow) != len(times) or len(inlet_flow) != len(times):
                    raise ValueError(
                        "Unexpected number of CFP boundary-flow outputs in the "
                        "listing-file node table."
                    )

                qfix_nodes = np.asarray(
                    listing_results.get("qfix_nodes", []), dtype=int
                )
                direct_nodes = np.asarray(
                    listing_results.get("direct_recharge_nodes", []), dtype=int
                )
                spring_node_number = int(qfix_nodes[0]) if qfix_nodes.size else None
                inlet_node_number = int(direct_nodes[0]) if direct_nodes.size else None

                def _connected_tube_for_node(node_number: int | None) -> int | None:
                    if node_number is None:
                        return None
                    begin = np.asarray(
                        listing_results["tube_begin_nodes"], dtype=int
                    )
                    end = np.asarray(
                        listing_results["tube_end_nodes"], dtype=int
                    )
                    connected = np.where(
                        (begin == int(node_number)) | (end == int(node_number))
                    )[0]
                    if connected.size == 0:
                        return None
                    return int(
                        np.asarray(listing_results["tube_numbers"], dtype=int)[
                            connected[0]
                        ]
                    )

                # Kept as metadata for orientation only. Current-result boundary
                # discharge is no longer taken from these tube flows.
                spring_tube_number = _connected_tube_for_node(spring_node_number)
                inlet_tube_number = _connected_tube_for_node(inlet_node_number)

                # -----------------------------------------------------------------
                # 1.3.13 Cumulative whole-run water budget
                # -----------------------------------------------------------------
                budget: dict | None = None
                budget_error: str | None = None
                try:
                    budget = parse_cumulative_water_budget(
                        listing_file,
                        expected_times=times,
                        listing_data=listing_results,
                    )
                except Exception as exc:
                    budget_error = str(exc)

                # -----------------------------------------------------------------
                # 1.3.14 Head and flow diagnostics
                # -----------------------------------------------------------------
                diagnostics: dict | None = None
                diagnostics_error: str | None = None

                try:
                    diagnostics = build_head_diagnostics(
                        workspace=workspace,
                        expected_times=times,
                        delr=delr,
                        delc=delc,
                        n_rows=n_rows,
                        n_cols=n_cols,
                        listing_data=listing_results,
                    )
                except Exception as exc:
                    # Do not discard a successful boundary-flow simulation just
                    # because the binary head file could not be post-processed.
                    diagnostics_error = str(exc)

    return {
        "times": np.asarray(times, dtype=float),
        # ``flow`` remains the stored-run comparison alias for spring discharge.
        "flow": np.asarray(spring_flow, dtype=float),
        "spring_flow": np.asarray(spring_flow, dtype=float),
        # Backward-compatible alias: this is now the external DIRECT RECHARGE
        # boundary flux, not the adjacent inlet-tube discharge.
        "inlet_flow": np.asarray(inlet_flow, dtype=float),
        "direct_recharge_flow": np.asarray(inlet_flow, dtype=float),
        "spring_node_number": spring_node_number,
        "inlet_node_number": inlet_node_number,
        "spring_tube_number": spring_tube_number,
        "inlet_tube_number": inlet_tube_number,
        "budget": budget,
        "budget_error": budget_error,
        "diagnostics": diagnostics,
        "diagnostics_error": diagnostics_error,
    }


# =============================================================================
# 2. MODEL OUTPUT AND POST-PROCESSING
# =============================================================================
# -----------------------------------------------------------------------------
# 2.1 Cumulative water-budget parsing
# -----------------------------------------------------------------------------
def _parse_last_cumulative_budget_block(
    lines: list[str],
    marker: str,
    terms: tuple[str, ...],
    stop_markers: tuple[str, ...],
) -> dict:
    """Parse cumulative IN/OUT values from the final matching listing block."""
    starts = [i for i, line in enumerate(lines) if marker in line.upper()]
    if not starts:
        raise ValueError(f"Budget section '{marker}' was not found in the listing file.")

    term_pattern = "|".join(
        re.escape(term) for term in sorted(terms, key=len, reverse=True)
    )
    value_re = re.compile(
        rf"^\s*({term_pattern})\s*=\s*({_NUMBER})",
        re.IGNORECASE,
    )

    values = {"in": {}, "out": {}}
    direction: str | None = None
    for line in lines[starts[-1] + 1 :]:
        upper = line.upper()
        if any(stop in upper for stop in stop_markers):
            break

        stripped = line.strip().upper()
        if stripped.startswith("IN:"):
            direction = "in"
            continue
        if stripped.startswith("OUT:"):
            direction = "out"
            continue

        match = value_re.match(line)
        if match and direction is not None:
            values[direction][match.group(1).upper()] = _as_float(match.group(2))

    return values


def _integrate_step_rates(times: np.ndarray, rates: np.ndarray) -> float:
    """Integrate end-of-step rates over the model time discretization."""
    times = np.asarray(times, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if times.ndim != 1 or rates.ndim != 1 or len(times) != len(rates):
        raise ValueError("Budget time/rate arrays are inconsistent.")
    if len(times) == 0:
        return 0.0
    dt = np.diff(np.concatenate(([0.0], times)))
    if np.any(dt <= 0.0):
        raise ValueError("Model output times must increase strictly.")
    return float(np.sum(rates * dt))


def parse_cumulative_water_budget(
    listing_file: Path,
    expected_times: np.ndarray,
    listing_data: dict | None = None,
) -> dict:
    """Return the requested cumulative whole-run water-budget components.

    External CFP boundary fluxes are taken directly from the node-result table:
      * ``DIRECT RECHARGE`` is the sinkhole/direct conduit inflow;
      * negative ``QFIX`` is the spring/fixed-head outflow and is plotted positive.

    This distinction matters because tube Q represents flow *between* conduit
    nodes. Exchange with the matrix can therefore make the boundary flux differ
    slightly from the adjacent tube flow.

    Matrix diffuse recharge, storage change and matrix-boundary flow come from
    MODFLOW's final cumulative ``VOLUMETRIC BUDGET FOR ENTIRE MODEL`` block.
    Matrix-conduit exchange is internal to the combined system and is omitted
    from the external whole-system budget.
    """
    lines = listing_file.read_text(encoding="utf-8", errors="replace").splitlines()

    matrix = _parse_last_cumulative_budget_block(
        lines,
        MODEL_BUDGET_MARKER,
        ("STORAGE", "CONSTANT HEAD", "RECHARGE", "PIPES"),
        ("TIME SUMMARY",),
    )

    if listing_data is None:
        listing_data = parse_cfp_listing(listing_file, expected_times)

    def net_in(block: dict, term: str) -> float:
        term = term.upper()
        return float(block["in"].get(term, 0.0) - block["out"].get(term, 0.0))

    def net_out(block: dict, term: str) -> float:
        term = term.upper()
        return float(block["out"].get(term, 0.0) - block["in"].get(term, 0.0))

    diffuse_recharge = net_in(matrix, "RECHARGE")
    matrix_storage_change = net_out(matrix, "STORAGE")
    matrix_boundary_outflow = net_out(matrix, "CONSTANT HEAD")

    direct_rate = np.asarray(listing_data["direct_recharge_total"], dtype=float)
    spring_rate = np.asarray(listing_data["spring_outflow"], dtype=float)
    direct_recharge = _integrate_step_rates(expected_times, direct_rate)
    karst_conduit_outflow = _integrate_step_rates(expected_times, spring_rate)

    residual = (
        diffuse_recharge
        + direct_recharge
        - matrix_storage_change
        - matrix_boundary_outflow
        - karst_conduit_outflow
    )

    return {
        "diffuse_recharge": diffuse_recharge,
        "direct_recharge": direct_recharge,
        "matrix_storage_change": matrix_storage_change,
        "matrix_boundary_outflow": matrix_boundary_outflow,
        "karst_conduit_outflow": karst_conduit_outflow,
        "residual": residual,
        "matrix_raw": matrix,
        "direct_recharge_source": "CFP node table: DIRECT RECHARGE",
        "spring_outflow_source": "CFP node table: negative QFIX",
    }


# -----------------------------------------------------------------------------
# 2.2 CFP listing-file parsing
# -----------------------------------------------------------------------------
def _as_float(value: str) -> float:
    """Convert ordinary or Fortran D-notation text to float."""
    return float(value.replace("D", "E").replace("d", "e"))


def _parse_cfp_node_result_line(
    line: str,
) -> tuple[int, float, float, float, float, float] | None:
    """Parse one CFP node-result row.

    The node table contains an optional ``FIX`` token after NODE HEAD for a
    fixed-head conduit node. After removing that token, the relevant numeric
    columns are:

      0 NODE HEAD
      1 MATRIX HEAD
      2 EXCHANGE
      6 DIRECT RECHARGE
     12 QFIX
    """
    parts = line.split()
    if not parts:
        return None
    try:
        node_number = int(parts[0])
    except ValueError:
        return None

    tokens = parts[1:]
    if len(tokens) > 1 and tokens[1].upper() == "FIX":
        tokens = tokens[:1] + tokens[2:]

    values: list[float] = []
    for token in tokens:
        try:
            values.append(_as_float(token))
        except ValueError:
            return None

    if len(values) < 13:
        return None

    return (
        node_number,
        values[0],   # conduit/node head
        values[1],   # matrix head
        values[2],   # matrix-conduit exchange
        values[6],   # direct recharge
        values[12],  # QFIX
    )


def parse_cfp_listing(listing_file: Path, expected_times: np.ndarray) -> dict:
    """Parse CFP heads and flows from every ``RESULTS OF FLOW CALCULATION`` block.

    The parser keeps the physically distinct CFP quantities separate:

    * conduit/node head and co-located matrix head;
    * signed matrix-conduit ``EXCHANGE`` at each node;
    * ``DIRECT RECHARGE`` at each node (external conduit inflow);
    * ``QFIX`` at each node (fixed-head boundary flow; negative is spring outflow);
    * actual tube ``Q`` between each pair of connected conduit nodes.

    Tube Q is therefore *not* used as a substitute for sinkhole inflow or spring
    outflow. Those boundary fluxes come directly from the node table.
    """
    text_listing = listing_file.read_text(encoding="utf-8", errors="replace")
    lines = text_listing.splitlines()

    # -------------------------------------------------------------------------
    # 2.2.1 CFP node coordinates
    # -------------------------------------------------------------------------
    coordinates: list[tuple[int, int, int, int, float, float, float]] = []
    reading_coordinates = False

    for line in lines:
        if "NODE" in line.upper() and "COLUMN" in line.upper() and "ROW" in line.upper():
            reading_coordinates = True
            continue

        if reading_coordinates:
            match = _NODE_COORD_RE.match(line)
            if match:
                groups = match.groups()
                coordinates.append(
                    (
                        int(groups[0]),
                        int(groups[1]),
                        int(groups[2]),
                        int(groups[3]),
                        _as_float(groups[4]),
                        _as_float(groups[5]),
                        _as_float(groups[6]),
                    )
                )
            elif coordinates:
                break

    if not coordinates:
        raise ValueError("No CFP node-coordinate table was found in the listing file.")

    coordinates.sort(key=lambda row: row[0])
    node_numbers = np.asarray([r[0] for r in coordinates], dtype=int)
    node_columns = np.asarray([r[1] for r in coordinates], dtype=int)
    node_rows = np.asarray([r[2] for r in coordinates], dtype=int)
    node_layers = np.asarray([r[3] for r in coordinates], dtype=int)
    node_x = np.asarray([r[4] for r in coordinates], dtype=float)
    node_y = np.asarray([r[5] for r in coordinates], dtype=float)
    node_z = np.asarray([r[6] for r in coordinates], dtype=float)

    if len(np.unique(node_numbers)) != len(node_numbers):
        raise ValueError("Duplicate CFP node numbers were found in the coordinate table.")

    # -------------------------------------------------------------------------
    # 2.2.2 CFP flow-calculation result blocks
    # -------------------------------------------------------------------------
    result_blocks: list[dict] = []
    current_stress_period: int | None = None
    current_time_step: int | None = None
    i = 0

    while i < len(lines):
        stress_match = _STRESS_TIME_RE.search(lines[i])
        if stress_match:
            current_stress_period = int(stress_match.group(1))
            current_time_step = int(stress_match.group(2))

        if FLOW_RESULTS_MARKER in lines[i].upper():
            i += 1
            while i < len(lines) and "NODE#" not in lines[i].upper():
                i += 1
            if i >= len(lines):
                raise ValueError(
                    "A 'RESULTS OF FLOW CALCULATION' section has no NODE# header."
                )

            i += 1
            node_rows_result: list[
                tuple[int, float, float, float, float, float]
            ] = []
            while i < len(lines):
                parsed = _parse_cfp_node_result_line(lines[i])
                if parsed is not None:
                    node_rows_result.append(parsed)
                    i += 1
                    continue
                if node_rows_result:
                    break
                i += 1

            if not node_rows_result:
                raise ValueError(
                    "A CFP flow-calculation section was found but no node results "
                    "could be parsed from it."
                )

            # Find and parse the tube-flow table belonging to the same output.
            while i < len(lines):
                upper = lines[i].upper()
                if "TUBE" in upper and "Q" in upper and "M^3/SEC" in upper:
                    break
                if FLOW_RESULTS_MARKER in upper:
                    raise ValueError(
                        "A CFP result block did not contain the expected tube-flow table."
                    )
                i += 1

            if i >= len(lines):
                raise ValueError(
                    "A CFP result block did not contain the expected tube-flow table."
                )

            i += 1
            tube_rows_result: list[tuple[int, int, int, float]] = []
            while i < len(lines):
                match = _TUBE_RESULT_RE.match(lines[i])
                if match:
                    tube_rows_result.append(
                        (
                            int(match.group(1)),
                            int(match.group(2)),
                            int(match.group(3)),
                            _as_float(match.group(4)),
                        )
                    )
                    i += 1
                    continue
                if tube_rows_result:
                    break
                i += 1

            if not tube_rows_result:
                raise ValueError(
                    "A CFP result block was found but no tube-flow rows could be parsed."
                )

            node_rows_result.sort(key=lambda row: row[0])
            tube_rows_result.sort(key=lambda row: row[0])
            result_blocks.append(
                {
                    "stress_period": current_stress_period,
                    "time_step": current_time_step,
                    "node_rows": node_rows_result,
                    "tube_rows": tube_rows_result,
                }
            )

        i += 1

    if not result_blocks:
        raise ValueError(
            "No 'RESULTS OF FLOW CALCULATION' blocks were found in the listing file."
        )

    if len(result_blocks) != len(expected_times):
        raise ValueError(
            "The listing file contains a different number of CFP result blocks "
            f"({len(result_blocks)}) than expected model times ({len(expected_times)})."
        )

    expected_node_list = node_numbers.tolist()
    first_tube_rows = result_blocks[0]["tube_rows"]
    tube_numbers = np.asarray([r[0] for r in first_tube_rows], dtype=int)
    tube_begin_nodes = np.asarray([r[1] for r in first_tube_rows], dtype=int)
    tube_end_nodes = np.asarray([r[2] for r in first_tube_rows], dtype=int)
    expected_tube_list = tube_numbers.tolist()

    n_time = len(result_blocks)
    n_node = len(node_numbers)
    conduit_heads = np.empty((n_time, n_node), dtype=float)
    matrix_heads_at_nodes = np.empty_like(conduit_heads)
    exchange_flow = np.empty_like(conduit_heads)
    direct_recharge = np.empty_like(conduit_heads)
    qfix = np.empty_like(conduit_heads)
    tube_flow = np.empty((n_time, len(tube_numbers)), dtype=float)
    stress_periods = np.empty(n_time, dtype=int)
    time_steps = np.empty(n_time, dtype=int)

    for block_idx, block in enumerate(result_blocks):
        node_rows_result = block["node_rows"]
        block_nodes = [r[0] for r in node_rows_result]
        if block_nodes != expected_node_list:
            raise ValueError(
                "The node sequence in a CFP result block does not match the node "
                "coordinate table."
            )

        tube_rows_result = block["tube_rows"]
        block_tubes = [r[0] for r in tube_rows_result]
        if block_tubes != expected_tube_list:
            raise ValueError("The tube sequence changes between CFP result blocks.")
        if [r[1] for r in tube_rows_result] != tube_begin_nodes.tolist() or [
            r[2] for r in tube_rows_result
        ] != tube_end_nodes.tolist():
            raise ValueError("The CFP tube connectivity changes between result blocks.")

        conduit_heads[block_idx, :] = [r[1] for r in node_rows_result]
        matrix_heads_at_nodes[block_idx, :] = [r[2] for r in node_rows_result]
        exchange_flow[block_idx, :] = [r[3] for r in node_rows_result]
        direct_recharge[block_idx, :] = [r[4] for r in node_rows_result]
        qfix[block_idx, :] = [r[5] for r in node_rows_result]
        tube_flow[block_idx, :] = [r[3] for r in tube_rows_result]
        stress_periods[block_idx] = int(block["stress_period"] or -1)
        time_steps[block_idx] = int(block["time_step"] or -1)

    # External conduit boundary fluxes. DIRECT RECHARGE is positive inflow.
    # Negative QFIX denotes fixed-head outflow and is converted to positive
    # spring discharge for user-facing plots.
    direct_recharge_total = np.sum(np.clip(direct_recharge, 0.0, None), axis=1)
    spring_outflow = np.sum(np.clip(-qfix, 0.0, None), axis=1)

    # Keep a representative node-flow magnitude for backward compatibility with
    # older diagnostic-comparison code, while preserving actual tube Q separately.
    node_conduit_flow = np.empty_like(conduit_heads)
    for node_idx, node_number in enumerate(node_numbers):
        connected = np.where(
            (tube_begin_nodes == node_number) | (tube_end_nodes == node_number)
        )[0]
        if connected.size == 0:
            raise ValueError(f"CFP node {node_number} is not connected to any tube.")
        node_conduit_flow[:, node_idx] = np.mean(
            np.abs(tube_flow[:, connected]), axis=1
        )

    node_index_by_number = {int(n): idx for idx, n in enumerate(node_numbers)}
    tube_mid_x = np.asarray(
        [
            0.5
            * (
                node_x[node_index_by_number[int(begin)]]
                + node_x[node_index_by_number[int(end)]]
            )
            for begin, end in zip(tube_begin_nodes, tube_end_nodes)
        ],
        dtype=float,
    )
    tube_mid_y = np.asarray(
        [
            0.5
            * (
                node_y[node_index_by_number[int(begin)]]
                + node_y[node_index_by_number[int(end)]]
            )
            for begin, end in zip(tube_begin_nodes, tube_end_nodes)
        ],
        dtype=float,
    )

    # Identify the actual boundary nodes from the node-table terms.
    direct_activity = np.max(np.abs(direct_recharge), axis=0)
    qfix_activity = np.max(np.abs(qfix), axis=0)
    direct_nodes = node_numbers[direct_activity > 0.0]
    qfix_nodes = node_numbers[qfix_activity > 0.0]

    return {
        "times": np.asarray(expected_times, dtype=float).copy(),
        "stress_periods": stress_periods,
        "time_steps": time_steps,
        "node_numbers": node_numbers,
        "node_columns": node_columns,
        "node_rows": node_rows,
        "node_layers": node_layers,
        "node_x": node_x,
        "node_y": node_y,
        "node_z": node_z,
        "conduit_heads": conduit_heads,
        "matrix_heads_at_nodes": matrix_heads_at_nodes,
        "exchange_flow": exchange_flow,
        "direct_recharge": direct_recharge,
        "qfix": qfix,
        "direct_recharge_total": direct_recharge_total,
        "spring_outflow": spring_outflow,
        "direct_recharge_nodes": direct_nodes,
        "qfix_nodes": qfix_nodes,
        "tube_numbers": tube_numbers,
        "tube_begin_nodes": tube_begin_nodes,
        "tube_end_nodes": tube_end_nodes,
        "tube_flow": tube_flow,
        "tube_flow_magnitude": np.abs(tube_flow),
        "tube_mid_x": tube_mid_x,
        "tube_mid_y": tube_mid_y,
        "node_conduit_flow": node_conduit_flow,
    }


# -----------------------------------------------------------------------------
# 2.3 MODFLOW matrix-head output and diagnostic assembly
# -----------------------------------------------------------------------------
def read_matrix_head_snapshots(
    head_file: Path,
    expected_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and align MODFLOW matrix heads to the CFP output times."""
    hds = bf.HeadFile(str(head_file))
    hds_times = np.asarray(hds.get_times(), dtype=float)

    if hds_times.size == 0:
        raise ValueError("The MODFLOW head file contains no saved times.")

    all_heads = np.asarray(hds.get_alldata(), dtype=np.float32)

    # Typical shape is [time, layer, row, column]. This model has one layer.
    if all_heads.ndim == 4:
        if all_heads.shape[1] < 1:
            raise ValueError("The MODFLOW head file contains no layers.")
        all_heads = all_heads[:, 0, :, :]
    elif all_heads.ndim != 3:
        raise ValueError(
            f"Unexpected MODFLOW head-array shape: {all_heads.shape}."
        )

    if all_heads.shape[0] != len(hds_times):
        raise ValueError(
            "The number of arrays in the MODFLOW head file does not match its "
            "reported output times."
        )

    expected_times = np.asarray(expected_times, dtype=float)
    aligned_indices: list[int] = []
    max_difference = 0.0

    for target_time in expected_times:
        idx = int(np.argmin(np.abs(hds_times - target_time)))
        difference = abs(float(hds_times[idx] - target_time))
        max_difference = max(max_difference, difference)
        aligned_indices.append(idx)

    # CFP/MODFLOW times in this model are integer seconds. A 1e-3 s tolerance
    # is intentionally generous relative to binary floating-point round-off.
    if max_difference > 1e-3:
        raise ValueError(
            "MODFLOW head times could not be aligned to CFP output times. "
            f"Maximum mismatch: {max_difference:.6g} s."
        )

    aligned_heads = all_heads[np.asarray(aligned_indices, dtype=int), :, :]
    return expected_times.copy(), aligned_heads


def build_head_diagnostics(
    workspace: Path,
    expected_times: np.ndarray,
    delr: float,
    delc: np.ndarray,
    n_rows: int,
    n_cols: int,
    listing_data: dict | None = None,
) -> dict:
    """Build all spatial head data needed by the optional result plots."""
    listing_file = find_output_file(
        workspace,
        f"{MODEL_NAME}.list",
        (".list", ".lst"),
    )
    head_file = find_output_file(
        workspace,
        f"{MODEL_NAME}.hds",
        (".hds", ".hed", ".head"),
    )

    listing = (
        listing_data
        if listing_data is not None
        else parse_cfp_listing(listing_file, expected_times)
    )
    head_times, matrix_heads = read_matrix_head_snapshots(head_file, expected_times)

    if matrix_heads.shape[1:] != (n_rows, n_cols):
        raise ValueError(
            "Unexpected MODFLOW head-grid shape. "
            f"Expected {(n_rows, n_cols)}, got {matrix_heads.shape[1:]}."
        )

    x_centers = (np.arange(n_cols, dtype=float) + 0.5) * float(delr)
    y_centers = np.cumsum(delc, dtype=float) - 0.5 * np.asarray(delc, dtype=float)

    # Cross-check the listing-file matrix heads against the corresponding cells
    # in the binary MODFLOW head file. This is a useful parser/mapping safeguard.
    row_idx = listing["node_rows"] - 1
    col_idx = listing["node_columns"] - 1

    if np.any(row_idx < 0) or np.any(row_idx >= n_rows):
        raise ValueError("A CFP node row lies outside the MODFLOW grid.")
    if np.any(col_idx < 0) or np.any(col_idx >= n_cols):
        raise ValueError("A CFP node column lies outside the MODFLOW grid.")

    binary_heads_at_nodes = matrix_heads[:, row_idx, col_idx]
    differences = np.abs(binary_heads_at_nodes - listing["matrix_heads_at_nodes"])
    finite_differences = differences[np.isfinite(differences)]
    max_matrix_difference = (
        float(np.max(finite_differences)) if finite_differences.size else np.nan
    )

    # Listing values are printed with limited precision. Larger differences are
    # therefore more likely to indicate a mapping or output-time mismatch.
    if np.isfinite(max_matrix_difference) and max_matrix_difference > 1e-2:
        raise ValueError(
            "The matrix heads parsed from the CFP listing file do not agree with "
            "the MODFLOW binary head file at conduit-node cells. Maximum absolute "
            f"difference: {max_matrix_difference:.4g} m."
        )

    return {
        **listing,
        "head_times": head_times,
        "matrix_heads": matrix_heads.astype(np.float32, copy=False),
        "x_centers": x_centers,
        "y_centers": y_centers,
        "matrix_head_consistency_max_abs_diff": max_matrix_difference,
    }


# =============================================================================
# 3. USER INPUT, RUN STATE, AND DIAGNOSTIC SELECTION
# =============================================================================
# -----------------------------------------------------------------------------
# 3.1 Synchronized numerical-input helpers
# -----------------------------------------------------------------------------
def _clip_numeric(value: float, minimum: float, maximum: float) -> float:
    """Clip a numerical UI value to the widget range."""
    return min(max(float(value), float(minimum)), float(maximum))


def _sync_linear_widget_to_value(
    widget_key: str, value_key: str, integer: bool = False
) -> None:
    """Copy a linear slider/number widget value into canonical session state."""
    value = st.session_state[widget_key]
    st.session_state[value_key] = int(value) if integer else float(value)


def synced_numeric_input(
    label: str,
    *,
    base_key: str,
    minimum: float,
    maximum: float,
    default: float,
    step: float,
    number_mode: bool,
    format_string: str | None = None,
    help_text: str | None = None,
    disabled: bool = False,
    integer: bool = False,
    force_sync: bool = False,
) -> float | int:
    """Render slider or number_input while preserving one canonical value.

    The widget type can be changed globally without changing the represented
    model parameter. A small per-control mode flag prevents stale inactive
    widget state from being restored when the user switches input mode.
    """
    value_key = f"{base_key}__value"
    slider_key = f"{base_key}__slider"
    number_key = f"{base_key}__number"
    mode_key = f"{base_key}__last_number_mode"

    if value_key not in st.session_state:
        initial = _clip_numeric(default, minimum, maximum)
        st.session_state[value_key] = int(round(initial)) if integer else float(initial)

    canonical = _clip_numeric(st.session_state[value_key], minimum, maximum)
    canonical = int(round(canonical)) if integer else float(canonical)
    st.session_state[value_key] = canonical

    previous_mode = st.session_state.get(mode_key)
    mode_changed = previous_mode is None or bool(previous_mode) != bool(number_mode)

    if number_mode:
        if force_sync or mode_changed or number_key not in st.session_state:
            st.session_state[number_key] = canonical
        number_kwargs = {
            "min_value": int(minimum) if integer else float(minimum),
            "max_value": int(maximum) if integer else float(maximum),
            "step": int(step) if integer else float(step),
            "key": number_key,
            "help": help_text,
            "disabled": disabled,
            "on_change": _sync_linear_widget_to_value,
            "args": (number_key, value_key, integer),
        }
        if format_string is not None:
            number_kwargs["format"] = format_string
        widget_value = st.number_input(label, **number_kwargs)
    else:
        if force_sync or mode_changed or slider_key not in st.session_state:
            st.session_state[slider_key] = canonical
        slider_kwargs = {
            "min_value": int(minimum) if integer else float(minimum),
            "max_value": int(maximum) if integer else float(maximum),
            "step": int(step) if integer else float(step),
            "key": slider_key,
            "help": help_text,
            "disabled": disabled,
            "on_change": _sync_linear_widget_to_value,
            "args": (slider_key, value_key, integer),
        }
        if format_string is not None:
            slider_kwargs["format"] = format_string
        widget_value = st.slider(label, **slider_kwargs)

    canonical = int(widget_value) if integer else float(widget_value)
    st.session_state[value_key] = canonical
    st.session_state[mode_key] = bool(number_mode)
    return canonical


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
    """
    Parameter input that can switch between slider and number input
    while preserving the current value.

    scale="linear":
        Standard linear slider.

    scale="log":
        Logarithmically spaced slider using physical parameter values.
        The corresponding number input uses an automatically scaled
        additive step.
    """

    import numpy as np
    import streamlit as st

    # -------------------------------------------------------------------------
    # Basic checks
    # -------------------------------------------------------------------------

    if scale not in ("linear", "log"):
        raise ValueError("scale must be 'linear' or 'log'.")

    if min_value >= max_value:
        raise ValueError("min_value must be smaller than max_value.")

    if not min_value <= default <= max_value:
        raise ValueError("default must be between min_value and max_value.")

    if scale == "log" and min_value <= 0:
        raise ValueError("Logarithmic parameters must be positive.")

    # -------------------------------------------------------------------------
    # Keys
    # -------------------------------------------------------------------------

    value_key = f"{key}__value"

    # -------------------------------------------------------------------------
    # Internal callback
    # -------------------------------------------------------------------------

    def update_value(widget_key):
        st.session_state[value_key] = float(
            st.session_state[widget_key]
        )

    # -------------------------------------------------------------------------
    # Permanent parameter state
    # -------------------------------------------------------------------------

    if value_key not in st.session_state:
        st.session_state[value_key] = float(default)

    current_value = float(st.session_state[value_key])

    current_value = min(
        max(current_value, float(min_value)),
        float(max_value),
    )

    st.session_state[value_key] = current_value

    # =========================================================================
    # NUMBER INPUT
    # =========================================================================

    if use_number_input:

        widget_key = f"_{key}__number"

        st.session_state[widget_key] = current_value

        # Determine suitable step
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

    # =========================================================================
    # LINEAR SLIDER
    # =========================================================================

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

    # =========================================================================
    # LOGARITHMIC SLIDER
    # =========================================================================

    else:

        widget_key = f"_{key}__slider"

        decades = np.log10(max_value) - np.log10(min_value)

        n_intervals = max(
            1,
            int(round(decades * log_steps_per_decade)),
        )

        options = np.logspace(
            np.log10(min_value),
            np.log10(max_value),
            n_intervals + 1,
        )

        # Preserve arbitrary values entered with number_input
        if not np.any(
            np.isclose(
                options,
                current_value,
                rtol=1e-12,
                atol=0.0,
            )
        ):
            options = np.append(options, current_value)

        options = np.unique(
            np.sort(options)
        ).tolist()

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


# -----------------------------------------------------------------------------
# 3.2 Stored-run data and rolling history
# -----------------------------------------------------------------------------
def parameter_table(params: dict) -> pd.DataFrame:
    labels = {
        "dmt": "Conduit diameter d [m]",
        "trtst": "Tortuosity [-]",
        "rh": "Roughness height [m]",
        "lcrey": "Lower critical Re [-]",
        "hcrey": "Higher critical Re [-]",
        "kxch": "Conduit wall permeability [m/s]",
        "cfptemp": "Water temperature [°C]",
        "hk": "Matrix hydraulic conductivity [m/s]",
        "sy": "Matrix specific yield [-]",
        "CADS": "CADS",
        "laminar_only": "Laminar-only mode",
    }
    return pd.DataFrame(
        {
            "Parameter": [labels[k] for k in params],
            "Value": [params[k] for k in params],
        }
    )


def parameter_sets_equal(first: dict, second: dict) -> bool:
    """Return True when two model-control dictionaries represent the same setup."""
    if first.keys() != second.keys():
        return False

    for key in first:
        a = first[key]
        b = second[key]
        if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            if not np.isclose(float(a), float(b), rtol=1.0e-12, atol=0.0):
                return False
        elif a != b:
            return False
    return True


def store_run_in_rolling_history(run: dict) -> dict:
    """Store a successful run in five cyclic, stable storage slots.

    Visible names are user-editable. Slot identity is therefore kept separately
    from the name so renaming a run can never break the rolling-history logic.
    """
    st.session_state.total_run_count += 1
    execution_number = int(st.session_state.total_run_count)
    slot_number = ((execution_number - 1) % MAX_STORED_RUNS) + 1
    default_name = f"run{slot_number}"

    stored_run = {
        "name": default_name,
        "slot_number": slot_number,
        "color": RUN_COLORS[slot_number - 1],
        "execution_number": execution_number,
        **run,
    }

    # Remove the previous content of this storage slot, independent of any
    # user-assigned display name, and append the new result as the newest run.
    history = [
        item
        for item in st.session_state.saved_scenarios
        if int(item["slot_number"]) != slot_number
    ]
    history.append(stored_run)
    st.session_state.saved_scenarios = history[-MAX_STORED_RUNS:]

    # Comparison selection uses stable execution numbers, not editable names.
    if "comparison_run_selection" in st.session_state:
        valid_ids = {
            int(item["execution_number"])
            for item in st.session_state.saved_scenarios
        }
        selected = [
            int(run_id)
            for run_id in st.session_state.comparison_run_selection
            if int(run_id) in valid_ids and int(run_id) != execution_number
        ]
        selected.append(execution_number)
        st.session_state.comparison_run_selection = selected

    return stored_run


def sync_current_run_name_from_widget() -> None:
    """Persist an edited current-run name before another model run starts.

    This helper is intentionally called before the Run button is processed. It
    therefore also catches a text edit when the user immediately clicks Run.
    """
    current = st.session_state.current_run
    if current is None:
        return

    key = f"run_name_input_{int(current['execution_number'])}"
    if key not in st.session_state:
        return

    proposed = str(st.session_state[key]).strip()
    if not proposed:
        proposed = f"run{int(current['slot_number'])}"

    current["name"] = proposed
    for item in st.session_state.saved_scenarios:
        if int(item["execution_number"]) == int(current["execution_number"]):
            item["name"] = proposed
            break


# -----------------------------------------------------------------------------
# 3.3 Diagnostic node/tube selection and geometry
# -----------------------------------------------------------------------------
def selected_node_metadata(diagnostics: dict, node_number: int) -> tuple[int, int]:
    """Return zero-based node index and one-based MODFLOW column for a CFP node."""
    node_numbers = np.asarray(diagnostics["node_numbers"], dtype=int)
    matches = np.where(node_numbers == int(node_number))[0]
    if matches.size != 1:
        raise ValueError(f"CFP node {node_number} could not be identified uniquely.")
    node_idx = int(matches[0])
    conduit_column = int(diagnostics["node_columns"][node_idx])
    return node_idx, conduit_column


def selected_tube_metadata(
    diagnostics: dict,
    tube_number: int,
) -> tuple[int, int, int, int, int]:
    """Return tube index plus begin/end node indices and node numbers."""
    tube_numbers = np.asarray(diagnostics["tube_numbers"], dtype=int)
    matches = np.where(tube_numbers == int(tube_number))[0]
    if matches.size != 1:
        raise ValueError(f"CFP tube {tube_number} could not be identified uniquely.")
    tube_idx = int(matches[0])
    begin_node = int(np.asarray(diagnostics["tube_begin_nodes"], dtype=int)[tube_idx])
    end_node = int(np.asarray(diagnostics["tube_end_nodes"], dtype=int)[tube_idx])
    begin_idx, _ = selected_node_metadata(diagnostics, begin_node)
    end_idx, _ = selected_node_metadata(diagnostics, end_node)
    return tube_idx, begin_idx, end_idx, begin_node, end_node


def adjacent_tubes_for_node(diagnostics: dict, node_number: int) -> dict[str, int]:
    """Return the left/right tube connected to a conduit node.

    ``Left`` and ``Right`` are defined geometrically from the x coordinate of the
    other end node, so the UI remains correct even if CFP tube numbering changes.
    End nodes naturally expose only one option.
    """
    node_idx, _ = selected_node_metadata(diagnostics, int(node_number))
    node_x = np.asarray(diagnostics["node_x"], dtype=float)
    x0 = float(node_x[node_idx])
    begin = np.asarray(diagnostics["tube_begin_nodes"], dtype=int)
    end = np.asarray(diagnostics["tube_end_nodes"], dtype=int)
    tube_numbers = np.asarray(diagnostics["tube_numbers"], dtype=int)

    connected = np.where((begin == int(node_number)) | (end == int(node_number)))[0]
    if connected.size == 0:
        raise ValueError(f"CFP node {node_number} is not connected to a conduit tube.")

    choices: dict[str, int] = {}
    for tube_idx in connected:
        b = int(begin[tube_idx])
        e = int(end[tube_idx])
        other_node = e if b == int(node_number) else b
        other_idx, _ = selected_node_metadata(diagnostics, other_node)
        other_x = float(node_x[other_idx])
        if other_x < x0:
            choices["Left"] = int(tube_numbers[tube_idx])
        elif other_x > x0:
            choices["Right"] = int(tube_numbers[tube_idx])
        else:
            # The teaching network is horizontal, but retain a deterministic
            # fallback in case a future network contains equal-x connected nodes.
            key = "Left" if other_node < int(node_number) else "Right"
            choices[key] = int(tube_numbers[tube_idx])

    # Keep the intuitive order in radio buttons.
    return {side: choices[side] for side in ("Left", "Right") if side in choices}


def default_tube_side(diagnostics: dict, node_number: int) -> str:
    """Choose a stable default adjacent-tube side for a node."""
    choices = adjacent_tubes_for_node(diagnostics, int(node_number))
    if "Right" in choices:
        return "Right"
    return next(iter(choices))


def _tube_profile_geometry(
    diagnostics: dict,
    tube_number: int,
) -> tuple[float, float, int, int]:
    """Return distances of the two nodes connected by a selected tube."""
    _, begin_idx, end_idx, begin_node, end_node = selected_tube_metadata(
        diagnostics, tube_number
    )
    node_x = np.asarray(diagnostics["node_x"], dtype=float)
    distance = node_x - node_x[0]
    return (
        float(distance[begin_idx]),
        float(distance[end_idx]),
        begin_node,
        end_node,
    )


# =============================================================================
# 4. PLOTTING AND DIAGNOSTIC VISUALIZATION
# =============================================================================
# -----------------------------------------------------------------------------
# 4.1 Common plotting, scale, and formatting helpers
# -----------------------------------------------------------------------------
def format_elapsed_time(seconds: float) -> str:
    """Compact time label for profile controls."""
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.2f} h"


def _finite_head_values(values: np.ndarray) -> np.ndarray:
    """Return finite, physically usable head values from a MODFLOW/CFP array."""
    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array) & (array > -1.0e20) & (array < 1.0e20)
    return array[valid]


def global_matrix_head_range(diagnostics: dict) -> tuple[float, float]:
    """Return the full-run matrix-head range used to initialize plan-view limits."""
    values = _finite_head_values(diagnostics["matrix_heads"])
    if values.size == 0:
        raise ValueError("The matrix-head results contain no valid values.")

    lower = float(np.min(values))
    upper = float(np.max(values))
    scale = max(abs(lower), abs(upper), 1.0)
    if upper - lower <= 1.0e-9 * scale:
        half_width = max(1.0e-3, 1.0e-4 * scale)
        lower -= half_width
        upper += half_width
    return lower, upper


def global_head_axis_range(diagnostics: dict) -> tuple[float, float]:
    """Return the full-run head range used to initialize profile/time-series axes."""
    parts = [
        _finite_head_values(diagnostics["matrix_heads"]),
        _finite_head_values(diagnostics["conduit_heads"]),
        _finite_head_values(diagnostics["matrix_heads_at_nodes"]),
    ]
    nonempty = [part for part in parts if part.size]
    if not nonempty:
        raise ValueError("The head diagnostics contain no valid values.")

    values = np.concatenate(nonempty)
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    scale = max(abs(lower), abs(upper), 1.0)
    padding = max(0.03 * span, 1.0e-3, 1.0e-4 * scale)
    return lower - padding, upper + padding


def complementary_color(color: str) -> str:
    """Return a visible complementary ring color for a run-specific conduit color."""
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    complement = 1.0 - rgb

    # A neutral gray has an almost identical RGB complement. In that special
    # case use a high-contrast gold ring so the selected node remains obvious.
    if float(np.linalg.norm(complement - rgb)) < 0.45:
        luminance = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
        return "#ffd700" if luminance < 0.65 else "#6a00ff"

    return mcolors.to_hex(complement)


def _nice_numeric_step(span: float, target_intervals: int = 40) -> float:
    """Choose a human-friendly slider step for a numeric range."""
    span = float(abs(span))
    if not np.isfinite(span) or span <= 0.0:
        return 0.1

    raw = span / max(int(target_intervals), 1)
    exponent = np.floor(np.log10(raw))
    fraction = raw / (10.0 ** exponent)
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return float(nice_fraction * (10.0 ** exponent))


def head_ceiling_slider_settings(
    lower_reference: float,
    observed_maximum: float,
) -> tuple[float, float, float]:
    """Return minimum, default/maximum and step for a user-adjustable head ceiling.

    The default is rounded upward from the observed full-run maximum. The slider
    intentionally only allows smaller ceilings because its purpose is to zoom
    into lower-head details without changing the numerical results.
    """
    lower_reference = float(lower_reference)
    observed_maximum = float(observed_maximum)
    if not np.isfinite(lower_reference) or not np.isfinite(observed_maximum):
        raise ValueError("Head limits must be finite.")

    if observed_maximum <= lower_reference:
        observed_maximum = lower_reference + max(1.0e-3, 1.0e-4 * abs(lower_reference))

    span = observed_maximum - lower_reference
    step = _nice_numeric_step(span)
    default_max = float(np.ceil(observed_maximum / step) * step)
    min_ceiling = float(np.ceil((lower_reference + step) / step) * step)

    if min_ceiling >= default_max:
        min_ceiling = float(default_max - step)
    return min_ceiling, default_max, step


def _nice_contour_ticks(vmin: float, vmax: float, nbins: int = 7) -> np.ndarray:
    """Generate round, readable contour/colorbar values inside a fixed range."""
    locator = MaxNLocator(
        nbins=nbins,
        steps=[1, 2, 2.5, 5, 10],
        min_n_ticks=3,
    )
    ticks = np.asarray(locator.tick_values(float(vmin), float(vmax)), dtype=float)
    tol = max(abs(vmax - vmin), 1.0) * 1.0e-10
    ticks = ticks[(ticks >= vmin - tol) & (ticks <= vmax + tol)]
    ticks = np.unique(ticks)
    if ticks.size < 2:
        ticks = np.linspace(vmin, vmax, 3)
    return ticks


def _head_tick_decimals(ticks: np.ndarray) -> int:
    """Choose a compact number of decimals for head labels."""
    ticks = np.asarray(ticks, dtype=float)
    if ticks.size < 2:
        return 2
    diffs = np.diff(np.sort(np.unique(ticks)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 2
    step = float(np.min(diffs))
    if step >= 1.0:
        return 0
    if step >= 0.1:
        return 1
    if step >= 0.01:
        return 2
    return 3


def _finite_flow_values(values: np.ndarray) -> np.ndarray:
    """Return finite CFP flow values."""
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def global_exchange_flow_limit(diagnostics: dict) -> float:
    """Rounded symmetric full-run limit for signed matrix-conduit exchange flow."""
    values = _finite_flow_values(diagnostics["exchange_flow"])
    if values.size == 0:
        raise ValueError("The exchange-flow results contain no valid values.")
    observed = float(np.max(np.abs(values)))
    if observed <= 0.0:
        return 1.0e-12
    step = _nice_numeric_step(observed, target_intervals=8)
    return float(np.ceil(observed / step) * step)


def flow_ceiling_slider_settings(observed_maximum: float) -> tuple[float, float, float]:
    """Return minimum, default maximum and step for conduit-flow ceiling slider."""
    observed_maximum = float(observed_maximum)
    if not np.isfinite(observed_maximum) or observed_maximum <= 0.0:
        return 1.0e-8, 1.0e-7, 1.0e-8

    step = _nice_numeric_step(observed_maximum, target_intervals=40)
    default_max = float(np.ceil(observed_maximum / step) * step)
    # Let the user zoom substantially while preventing a zero-height axis.
    min_ceiling = float(max(step, default_max / 20.0))
    min_ceiling = float(np.ceil(min_ceiling / step) * step)
    if min_ceiling >= default_max:
        min_ceiling = step
    return min_ceiling, default_max, step


def _scenario_labels(scenarios: list[dict]) -> dict[int, str]:
    """Return unambiguous display labels keyed by execution number."""
    counts: dict[str, int] = {}
    for scenario in scenarios:
        counts[scenario["name"]] = counts.get(scenario["name"], 0) + 1
    labels: dict[int, str] = {}
    for scenario in scenarios:
        label = scenario["name"]
        if counts[label] > 1:
            label = f"{label} (execution {scenario['execution_number']})"
        labels[int(scenario["execution_number"])] = label
    return labels


def _nearest_diagnostic_time_index(diagnostics: dict, target_time: float) -> int:
    times = np.asarray(diagnostics["times"], dtype=float)
    return int(np.argmin(np.abs(times - float(target_time))))


def _conduit_flow_profile_data(
    diagnostics: dict,
    time_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outlet boundary, actual tube Q values, and inlet boundary as one profile."""
    node_x = np.asarray(diagnostics["node_x"], dtype=float)
    outlet_x = float(node_x[0])
    inlet_x = float(node_x[-1])
    tube_mid_x = np.asarray(diagnostics["tube_mid_x"], dtype=float)
    tube_q = np.abs(np.asarray(diagnostics["tube_flow"][time_index, :], dtype=float))
    spring_q = float(np.asarray(diagnostics["spring_outflow"], dtype=float)[time_index])
    direct_q = float(
        np.asarray(diagnostics["direct_recharge_total"], dtype=float)[time_index]
    )

    x = np.concatenate(([outlet_x], tube_mid_x, [inlet_x])) - outlet_x
    q = np.concatenate(([spring_q], tube_q, [direct_q]))
    return x, q


def global_conduit_flow_max(diagnostics: dict) -> float:
    """Maximum of actual tube flow and the two external conduit boundary fluxes."""
    arrays = [
        np.abs(np.asarray(diagnostics["tube_flow"], dtype=float)).ravel(),
        np.asarray(diagnostics["spring_outflow"], dtype=float).ravel(),
        np.asarray(diagnostics["direct_recharge_total"], dtype=float).ravel(),
    ]
    values = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    if values.size == 0:
        raise ValueError("The conduit-flow results contain no valid values.")
    return float(np.max(values))


# -----------------------------------------------------------------------------
# 4.2 Spring-response comparison
# -----------------------------------------------------------------------------
def make_comparison_plot(
    scenarios: list[dict],
    time_unit: str,
):
    """Plot automatically stored model runs for direct comparison."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    if time_unit == "hours":
        time_factor = 3600.0
        x_label = "Time [h]"
    else:
        time_factor = 1.0
        x_label = "Time [s]"

    # Duplicate user names are allowed because run identity is tracked by the
    # execution number. Add the execution number to duplicate legend labels so
    # the comparison remains unambiguous.
    name_counts: dict[str, int] = {}
    for scenario in scenarios:
        name_counts[scenario["name"]] = name_counts.get(scenario["name"], 0) + 1

    for scenario in scenarios:
        label = scenario["name"]
        if name_counts[label] > 1:
            label = f"{label} (execution {scenario['execution_number']})"
        ax.plot(
            scenario["times"] / time_factor,
            scenario["flow"],
            linewidth=2,
            color=scenario["color"],
            label=label,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Spring discharge [m³/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 4.3 Head diagnostics
# -----------------------------------------------------------------------------
def make_plan_view_head_plot(
    diagnostics: dict,
    time_index: int,
    conduit_color: str,
    selected_node: int,
    matrix_head_max: float,
):
    """Plot matrix-head contours and the shared selected conduit node/transect."""
    head = np.asarray(diagnostics["matrix_heads"][time_index, :, :], dtype=float)
    invalid = (~np.isfinite(head)) | (head <= -1.0e20) | (head >= 1.0e20)
    head_masked = np.ma.masked_where(invalid, head)
    valid = head_masked.compressed()

    if valid.size == 0:
        raise ValueError("The selected matrix-head field contains no valid values.")

    global_min, global_max = global_matrix_head_range(diagnostics)
    matrix_head_max = float(matrix_head_max)
    if matrix_head_max <= global_min:
        raise ValueError("The selected maximum matrix head must exceed the minimum head.")

    # The user may deliberately lower the ceiling to emphasize lower-head
    # differences. Values above it are saturated at the top color rather than
    # disappearing from the map.
    fill_levels = np.linspace(global_min, matrix_head_max, 21)
    line_levels = _nice_contour_ticks(global_min, matrix_head_max, nbins=8)
    decimals = _head_tick_decimals(line_levels)

    x = np.asarray(diagnostics["x_centers"], dtype=float)
    y = np.asarray(diagnostics["y_centers"], dtype=float)
    node_idx, conduit_column = selected_node_metadata(diagnostics, selected_node)
    ring_color = complementary_color(conduit_color)

    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    # ``x`` and ``y`` contain MODFLOW cell-center coordinates.  A contourf
    # drawn only on those centers naturally stops half a cell before the outer
    # model boundary.  Extend the *filled* field by one copied edge value on
    # each side so the color fill reaches the actual cell boundaries.  The
    # contour lines below still use the original cell-center head field.
    if x.size > 1:
        dx_fill = float(np.median(np.diff(x)))
    else:
        dx_fill = max(float(x[0]) * 2.0, 1.0)
    if y.size > 1:
        dy_fill = float(np.median(np.diff(y)))
    else:
        dy_fill = max(float(y[0]) * 2.0, 1.0)

    x_fill = np.concatenate(
        ([x[0] - 0.5 * dx_fill], x, [x[-1] + 0.5 * dx_fill])
    )
    y_fill = np.concatenate(
        ([y[0] - 0.5 * dy_fill], y, [y[-1] + 0.5 * dy_fill])
    )
    head_fill = np.pad(head, ((1, 1), (1, 1)), mode="edge")
    invalid_fill = np.pad(invalid, ((1, 1), (1, 1)), mode="edge")
    head_fill_masked = np.ma.masked_where(invalid_fill, head_fill)

    filled = ax.contourf(
        x_fill,
        y_fill,
        head_fill_masked,
        levels=fill_levels,
        cmap=HEAD_COLORMAP,
        extend="max" if matrix_head_max < global_max else "neither",
    )

    selected_min = float(np.min(valid))
    selected_max = float(np.max(valid))
    visible_line_levels = line_levels[
        (line_levels >= selected_min) & (line_levels <= selected_max)
    ]
    if visible_line_levels.size:
        lines = ax.contour(
            x,
            y,
            head_masked,
            levels=visible_line_levels,
            colors="black",
            linewidths=0.8,
            alpha=0.55,
        )
        ax.clabel(
            lines,
            inline=True,
            fontsize=8,
            fmt=lambda value: f"{value:.{decimals}f}",
        )

    ax.plot(
        diagnostics["node_x"],
        diagnostics["node_y"],
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        color=conduit_color,
        label="Conduit",
        zorder=5,
    )

    # The selected conduit node also defines the perpendicular section used in
    # section 2, so only one spatial selection is needed for all diagnostics.
    x_position = float(diagnostics["x_centers"][conduit_column - 1])
    ax.axvline(
        x_position,
        linestyle="--",
        linewidth=1.7,
        color=REFERENCE_COLOR,
        label=f"Perpendicular profile through node {selected_node}",
        zorder=4,
    )

    selected_x = float(diagnostics["node_x"][node_idx])
    selected_y = float(diagnostics["node_y"][node_idx])
    ax.scatter(
        [selected_x],
        [selected_y],
        s=62,
        marker="o",
        color=conduit_color,
        edgecolors="white",
        linewidths=0.8,
        zorder=7,
    )
    ax.scatter(
        [selected_x],
        [selected_y],
        s=175,
        marker="o",
        facecolors="none",
        edgecolors=ring_color,
        linewidths=2.6,
        label=f"Selected conduit node {selected_node}",
        zorder=8,
    )

    selected_time = float(diagnostics["times"][time_index])
    ax.set_title(f"Matrix head at t = {format_elapsed_time(selected_time)}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    # ``x`` contains cell centers (25, 75, ... m). Explicitly show the full
    # model extent so the first cell starts at x = 0 while conduit nodes remain
    # correctly positioned at their cell centers.
    if x.size > 1:
        dx = float(np.median(np.diff(x)))
    else:
        dx = max(float(x[0]) * 2.0, 1.0)
    ax.set_xlim(0.0, float(x[-1] + 0.5 * dx))
    ax.legend(loc="best")

    cbar = fig.colorbar(filled, ax=ax, pad=0.02)
    cbar.set_label("Matrix hydraulic head [m]")
    colorbar_ticks = _nice_contour_ticks(global_min, matrix_head_max, nbins=6)
    colorbar_decimals = _head_tick_decimals(colorbar_ticks)
    cbar.set_ticks(colorbar_ticks)
    cbar.set_ticklabels(
        [f"{value:.{colorbar_decimals}f}" for value in colorbar_ticks]
    )

    fig.tight_layout()
    return fig


def make_plan_view_head_plot_v6(
    diagnostics: dict,
    time_index: int,
    conduit_color: str,
    selected_node: int,
    matrix_head_max: float,
    comparison_node: int | None = None,
):
    """Use the established plan view and optionally mark a second comparison node."""
    fig = make_plan_view_head_plot(
        diagnostics,
        time_index,
        conduit_color=conduit_color,
        selected_node=selected_node,
        matrix_head_max=matrix_head_max,
    )
    if comparison_node is None or int(comparison_node) == int(selected_node):
        return fig

    ax = fig.axes[0]
    node_idx, conduit_column = selected_node_metadata(diagnostics, int(comparison_node))
    x_position = float(diagnostics["x_centers"][conduit_column - 1])
    selected_x = float(diagnostics["node_x"][node_idx])
    selected_y = float(diagnostics["node_y"][node_idx])

    ax.axvline(
        x_position,
        linestyle=":",
        linewidth=1.7,
        color=COMPARISON_REFERENCE_COLOR,
        alpha=0.9,
        label=f"Comparison section through node {comparison_node}",
        zorder=4,
    )
    ax.scatter(
        [selected_x],
        [selected_y],
        s=62,
        marker="o",
        color=COMPARISON_REFERENCE_COLOR,
        edgecolors="white",
        linewidths=0.8,
        zorder=8,
    )
    ax.scatter(
        [selected_x],
        [selected_y],
        s=175,
        marker="o",
        facecolors="none",
        edgecolors=COMPARISON_REFERENCE_COLOR,
        linewidths=2.4,
        label=f"Comparison node {comparison_node}",
        zorder=9,
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_plan_view_head_plot_v7(
    diagnostics: dict,
    time_index: int,
    conduit_color: str,
    selected_node: int,
    selected_tube: int,
    matrix_head_max: float,
    comparison_node: int | None = None,
    comparison_tube: int | None = None,
):
    """Plan view with node-centered tube selections."""
    fig = make_plan_view_head_plot_v6(
        diagnostics,
        time_index,
        conduit_color=conduit_color,
        selected_node=selected_node,
        matrix_head_max=matrix_head_max,
        comparison_node=comparison_node,
    )
    ax = fig.axes[0]
    _draw_tube_symbol_plan(
        ax,
        diagnostics,
        int(selected_tube),
        color=complementary_color(conduit_color),
        label=f"Selected tube {selected_tube}",
        selected_node=int(selected_node),
    )
    if comparison_tube is not None and int(comparison_tube) != int(selected_tube):
        _draw_tube_symbol_plan(
            ax,
            diagnostics,
            int(comparison_tube),
            color=COMPARISON_REFERENCE_COLOR,
            label=f"Comparison tube {comparison_tube}",
            selected_node=(int(comparison_node) if comparison_node is not None else None),
            linestyle="--",
            alpha=0.9,
        )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_longitudinal_head_plot_v6(
    diagnostics: dict,
    time_index: int,
    conduit_color: str,
    selected_node: int,
    head_ylim: tuple[float, float],
    comparison_time_index: int | None = None,
    comparison_node: int | None = None,
):
    """Plot longitudinal heads for a base selection and optional second time/node."""
    node_x = np.asarray(diagnostics["node_x"], dtype=float)
    distance = node_x - node_x[0]
    base_idx, _ = selected_node_metadata(diagnostics, selected_node)
    base_time = float(diagnostics["times"][time_index])
    ring_color = complementary_color(conduit_color)

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    base_con = np.asarray(diagnostics["conduit_heads"][time_index, :], dtype=float)
    base_mat = np.asarray(diagnostics["matrix_heads_at_nodes"][time_index, :], dtype=float)
    ax.plot(distance, base_con, linewidth=2.2, color=conduit_color,
            label=f"Conduit — {format_elapsed_time(base_time)}")
    ax.plot(distance, base_mat, linewidth=2.0, color=MATRIX_HEAD_COLOR,
            label=f"Matrix — {format_elapsed_time(base_time)}")

    base_x = float(distance[base_idx])
    ax.axvline(base_x, linestyle=":", linewidth=1.4, color=ring_color, alpha=0.9)
    ax.scatter([base_x], [float(base_con[base_idx])], s=62, color=conduit_color,
               edgecolors="white", linewidths=0.8, zorder=7)
    ax.scatter([base_x], [float(base_con[base_idx])], s=150, facecolors="none",
               edgecolors=ring_color, linewidths=2.4, label=f"Base node {selected_node}", zorder=8)

    if comparison_time_index is not None:
        compare_time = float(diagnostics["times"][comparison_time_index])
        compare_con = np.asarray(
            diagnostics["conduit_heads"][comparison_time_index, :], dtype=float
        )
        compare_mat = np.asarray(
            diagnostics["matrix_heads_at_nodes"][comparison_time_index, :], dtype=float
        )
        ax.plot(distance, compare_con, linewidth=2.2, linestyle="--", color=conduit_color,
                alpha=0.78, label=f"Conduit — {format_elapsed_time(compare_time)} (comparison)")
        ax.plot(distance, compare_mat, linewidth=2.0, linestyle="--", color=MATRIX_HEAD_COLOR,
                alpha=0.62, label=f"Matrix — {format_elapsed_time(compare_time)} (comparison)")

        compare_node_value = int(comparison_node if comparison_node is not None else selected_node)
        compare_idx, _ = selected_node_metadata(diagnostics, compare_node_value)
        compare_x = float(distance[compare_idx])
        ax.axvline(compare_x, linestyle="--", linewidth=1.2,
                   color=COMPARISON_REFERENCE_COLOR, alpha=0.8)
        ax.scatter([compare_x], [float(compare_con[compare_idx])], s=58, marker="o",
                   color=COMPARISON_REFERENCE_COLOR, edgecolors="white",
                   linewidths=0.8, zorder=7)
        ax.scatter([compare_x], [float(compare_con[compare_idx])], s=135, marker="o",
                   facecolors="none", edgecolors=COMPARISON_REFERENCE_COLOR,
                   linewidths=2.2, label=f"Comparison node {compare_node_value}", zorder=8)

    ax.set_title("Heads along the conduit")
    ax.set_xlabel("Distance along conduit from outlet [m]")
    ax.set_ylabel("Hydraulic head [m]")
    ax.set_ylim(*head_ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_perpendicular_head_plot_v6(
    diagnostics: dict,
    time_index: int,
    selected_node: int,
    conduit_color: str,
    head_ylim: tuple[float, float],
    comparison_time_index: int | None = None,
    comparison_node: int | None = None,
):
    """Plot one or two perpendicular matrix/conduit head sections."""
    base_idx, base_col = selected_node_metadata(diagnostics, selected_node)
    base_y = float(diagnostics["node_y"][base_idx])
    y_distance = np.asarray(diagnostics["y_centers"], dtype=float) - base_y
    base_profile = np.asarray(diagnostics["matrix_heads"][time_index, :, base_col - 1], dtype=float)
    base_conduit = float(diagnostics["conduit_heads"][time_index, base_idx])
    base_time = float(diagnostics["times"][time_index])
    ring_color = complementary_color(conduit_color)

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.plot(y_distance, base_profile, linewidth=2.0, color=MATRIX_HEAD_COLOR,
            label=f"Matrix — node {selected_node}, {format_elapsed_time(base_time)}")
    ax.scatter([0.0], [base_conduit], s=64, color=conduit_color,
               edgecolors="white", linewidths=0.8,
               label=f"Conduit — node {selected_node}", zorder=7)
    ax.scatter([0.0], [base_conduit], s=155, facecolors="none",
               edgecolors=ring_color, linewidths=2.4, zorder=8)

    if comparison_time_index is not None:
        compare_node_value = int(comparison_node if comparison_node is not None else selected_node)
        compare_idx, compare_col = selected_node_metadata(diagnostics, compare_node_value)
        compare_y = float(diagnostics["node_y"][compare_idx])
        compare_distance = np.asarray(diagnostics["y_centers"], dtype=float) - compare_y
        compare_profile = np.asarray(
            diagnostics["matrix_heads"][comparison_time_index, :, compare_col - 1], dtype=float
        )
        compare_conduit = float(
            diagnostics["conduit_heads"][comparison_time_index, compare_idx]
        )
        compare_time = float(diagnostics["times"][comparison_time_index])
        ax.plot(compare_distance, compare_profile, linewidth=2.0, linestyle="--",
                color=MATRIX_HEAD_COLOR, alpha=0.62,
                label=f"Matrix — node {compare_node_value}, {format_elapsed_time(compare_time)} (comparison)")
        ax.scatter([0.0], [compare_conduit], s=125, marker="D", facecolors="none",
                   edgecolors=COMPARISON_REFERENCE_COLOR, linewidths=2.2,
                   label=f"Conduit — comparison node {compare_node_value}", zorder=8)

    ax.axvline(0.0, linestyle=":", linewidth=1.2, color=REFERENCE_COLOR, alpha=0.8)
    ax.set_title("Perpendicular matrix-head profile")
    ax.set_xlabel("Distance perpendicular to conduit [m]")
    ax.set_ylabel("Hydraulic head [m]")
    ax.set_ylim(*head_ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_node_head_timeseries_plot_v6(
    diagnostics: dict,
    node_number: int,
    conduit_color: str,
    selected_time_index: int,
    head_ylim: tuple[float, float],
    comparison_node: int | None = None,
    comparison_time_index: int | None = None,
):
    """Plot transient heads at a base node and optional comparison node."""
    times_h = np.asarray(diagnostics["times"], dtype=float) / 3600.0
    base_idx, _ = selected_node_metadata(diagnostics, node_number)
    base_con = np.asarray(diagnostics["conduit_heads"][:, base_idx], dtype=float)
    base_mat = np.asarray(diagnostics["matrix_heads_at_nodes"][:, base_idx], dtype=float)
    ring_color = complementary_color(conduit_color)

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.plot(times_h, base_con, linewidth=2.2, color=conduit_color,
            label=f"Conduit — node {node_number}")
    ax.plot(times_h, base_mat, linewidth=2.0, color=MATRIX_HEAD_COLOR,
            label=f"Matrix — node {node_number}")

    base_t = float(times_h[selected_time_index])
    ax.axvline(base_t, linestyle=":", linewidth=1.4, color=ring_color, alpha=0.9,
               label="Base diagnostic time")
    ax.scatter([base_t], [float(base_con[selected_time_index])], s=140,
               facecolors="none", edgecolors=ring_color, linewidths=2.3, zorder=8)

    if comparison_node is not None:
        compare_idx, _ = selected_node_metadata(diagnostics, int(comparison_node))
        compare_con = np.asarray(diagnostics["conduit_heads"][:, compare_idx], dtype=float)
        compare_mat = np.asarray(diagnostics["matrix_heads_at_nodes"][:, compare_idx], dtype=float)
        ax.plot(times_h, compare_con, linewidth=2.2, linestyle="--", color=conduit_color,
                alpha=0.78, label=f"Conduit — node {comparison_node} (comparison)")
        ax.plot(times_h, compare_mat, linewidth=2.0, linestyle="--", color=MATRIX_HEAD_COLOR,
                alpha=0.62, label=f"Matrix — node {comparison_node} (comparison)")
        if comparison_time_index is not None:
            compare_t = float(times_h[comparison_time_index])
            ax.axvline(compare_t, linestyle="--", linewidth=1.2,
                       color=COMPARISON_REFERENCE_COLOR, alpha=0.8,
                       label="Comparison diagnostic time")
            ax.scatter([compare_t], [float(compare_con[comparison_time_index])], s=125,
                       marker="D", facecolors="none", edgecolors=COMPARISON_REFERENCE_COLOR,
                       linewidths=2.2, zorder=8)

    ax.set_title("Transient heads at selected conduit node(s)")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Hydraulic head [m]")
    ax.set_ylim(*head_ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_head_profile_comparison_plot(
    scenarios: list[dict], target_time: float, node_number: int
):
    """Compare longitudinal conduit and co-located matrix heads across runs."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = _scenario_labels(scenarios)
    all_values: list[np.ndarray] = []

    for scenario in scenarios:
        diagnostics = scenario["diagnostics"]
        idx = _nearest_diagnostic_time_index(diagnostics, target_time)
        node_x = np.asarray(diagnostics["node_x"], dtype=float)
        distance = node_x - node_x[0]
        conduit = np.asarray(diagnostics["conduit_heads"][idx, :], dtype=float)
        matrix = np.asarray(diagnostics["matrix_heads_at_nodes"][idx, :], dtype=float)
        label = labels[int(scenario["execution_number"])]
        ax.plot(distance, conduit, linewidth=2.1, color=scenario["color"], label=f"{label} — conduit")
        ax.plot(distance, matrix, linewidth=1.8, linestyle="--", color=scenario["color"], alpha=0.8, label=f"{label} — matrix")
        all_values.extend([conduit, matrix])

    # Shared selected-node marker by position; all model geometries are identical.
    first_d = scenarios[0]["diagnostics"]
    node_idx, _ = selected_node_metadata(first_d, node_number)
    first_dist = np.asarray(first_d["node_x"], dtype=float) - float(first_d["node_x"][0])
    ax.axvline(float(first_dist[node_idx]), linestyle=":", linewidth=1.3, color=REFERENCE_COLOR)
    ax.set_title(f"Stored runs — heads along conduit at t = {format_elapsed_time(target_time)}")
    ax.set_xlabel("Distance along conduit from outlet [m]")
    ax.set_ylabel("Hydraulic head [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def make_head_timeseries_comparison_plot(scenarios: list[dict], node_number: int):
    """Compare transient conduit/matrix heads at the selected node across runs."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = _scenario_labels(scenarios)
    for scenario in scenarios:
        diagnostics = scenario["diagnostics"]
        node_idx, _ = selected_node_metadata(diagnostics, node_number)
        times_h = np.asarray(diagnostics["times"], dtype=float) / 3600.0
        conduit = np.asarray(diagnostics["conduit_heads"][:, node_idx], dtype=float)
        matrix = np.asarray(diagnostics["matrix_heads_at_nodes"][:, node_idx], dtype=float)
        label = labels[int(scenario["execution_number"])]
        ax.plot(times_h, conduit, linewidth=2.1, color=scenario["color"], label=f"{label} — conduit")
        ax.plot(times_h, matrix, linewidth=1.8, linestyle="--", color=scenario["color"], alpha=0.8, label=f"{label} — matrix")
    ax.set_title(f"Stored runs — transient heads at conduit node {node_number}")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Hydraulic head [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 4.4 Flow diagnostics
# -----------------------------------------------------------------------------
def _draw_tube_symbol_plan(
    ax,
    diagnostics: dict,
    tube_number: int,
    color: str,
    label: str,
    selected_node: int | None = None,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 10,
):
    """Draw a selected tube as two parallel lines between its conduit nodes.

    When ``selected_node`` is supplied, that node is left to the parent plot's
    filled-node marker, while the opposite tube end is emphasized by an open
    circle. This gives one consistent visual grammar: filled circle = selected
    node, open circle = the other end of the selected tube.
    """
    _, begin_idx, end_idx, begin_node, end_node = selected_tube_metadata(
        diagnostics, tube_number
    )
    x1 = float(diagnostics["node_x"][begin_idx])
    y1 = float(diagnostics["node_y"][begin_idx])
    x2 = float(diagnostics["node_x"][end_idx])
    y2 = float(diagnostics["node_y"][end_idx])

    dx, dy = x2 - x1, y2 - y1
    length = max(float(np.hypot(dx, dy)), 1.0)
    offset = min(4.0, max(1.5, 0.04 * length))
    ox = -dy / length * offset
    oy = dx / length * offset

    ax.plot(
        [x1 + ox, x2 + ox], [y1 + oy, y2 + oy],
        color=color, linewidth=2.0, linestyle=linestyle, alpha=alpha,
        label=label, zorder=zorder,
    )
    ax.plot(
        [x1 - ox, x2 - ox], [y1 - oy, y2 - oy],
        color=color, linewidth=2.0, linestyle=linestyle, alpha=alpha,
        zorder=zorder,
    )

    if selected_node is not None and int(selected_node) in (begin_node, end_node):
        other_idx = end_idx if int(selected_node) == begin_node else begin_idx
        ax.scatter(
            [float(diagnostics["node_x"][other_idx])],
            [float(diagnostics["node_y"][other_idx])],
            s=58, marker="o", facecolors="white", edgecolors=color,
            linewidths=2.0, alpha=alpha, zorder=zorder + 2,
        )
    else:
        ax.scatter(
            [x1, x2], [y1, y2], s=44, marker="o", facecolors="white",
            edgecolors=color, linewidths=1.8, alpha=alpha, zorder=zorder + 1,
        )


def make_conduit_flow_profile_plot_v7(
    diagnostics: dict,
    time_index: int,
    conduit_color: str,
    selected_node: int,
    selected_tube: int,
    flow_max: float,
    comparison_time_index: int | None = None,
    comparison_node: int | None = None,
    comparison_tube: int | None = None,
):
    """Plot external boundary fluxes and actual tube Q along the conduit.

    The blue profile contains the external boundary fluxes at the two end nodes
    and actual CFP tube-flow values at tube midpoints.  Because tube Q is not a
    node quantity, the flow value at an internal conduit node is obtained by
    linear interpolation of this already plotted profile.  This lets the
    selected tube be highlighted *on the profile itself*:

    * circles mark the two tube-end nodes on the plotted flow profile;
    * the selected conduit node is filled and the other tube end is open;
    * a square marks the actual tube Q at the tube midpoint;
    * the colored highlight follows the same piecewise profile between the two
      node positions, rather than drawing a separate horizontal tube symbol.

    At node 1 and node N the profile uses the external CFP boundary fluxes
    (spring -QFIX and DIRECT RECHARGE respectively), exactly as in the main
    blue curve.
    """
    base_x, base_q = _conduit_flow_profile_data(diagnostics, time_index)
    base_time = float(diagnostics["times"][time_index])
    highlight_color = complementary_color(conduit_color)

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.plot(
        base_x, base_q, linewidth=2.2, marker="s", markersize=3.6,
        color=conduit_color,
        label=f"Boundary + tube flow — {format_elapsed_time(base_time)}",
    )

    def draw_active_tube(
        tube_number: int,
        node_number: int,
        t_index: int,
        profile_x: np.ndarray,
        profile_q: np.ndarray,
        color: str,
        label: str,
        linestyle: str = "-",
        alpha: float = 1.0,
    ) -> None:
        tube_idx, _, _, begin_node, end_node = selected_tube_metadata(
            diagnostics, int(tube_number)
        )
        x1, x2, _, _ = _tube_profile_geometry(diagnostics, int(tube_number))

        # Actual CFP tube flow is located at the tube midpoint in the main
        # profile.  The node marker elevations are read from that same plotted
        # profile by interpolation, so all highlighting lies exactly on it.
        q_tube = abs(float(diagnostics["tube_flow"][t_index, tube_idx]))
        x_mid = 0.5 * (x1 + x2)
        y1 = float(np.interp(x1, profile_x, profile_q))
        y2 = float(np.interp(x2, profile_x, profile_q))

        if int(node_number) == begin_node:
            selected_x, selected_y = x1, y1
            other_x, other_y = x2, y2
        elif int(node_number) == end_node:
            selected_x, selected_y = x2, y2
            other_x, other_y = x1, y1
        else:
            raise ValueError(
                f"Selected tube {tube_number} is not adjacent to node {node_number}."
            )

        # Follow the same piecewise line as the blue profile: node -> tube
        # midpoint -> node.  This is especially important at the sinkhole and
        # spring ends, where the external boundary flux can differ strongly
        # from the adjacent tube Q because of matrix-conduit exchange.
        ax.plot(
            [x1, x_mid, x2], [y1, q_tube, y2],
            color=color, linewidth=3.0, linestyle=linestyle,
            alpha=alpha, label=label, zorder=8,
        )

        # Tube quantity: square at the tube midpoint.
        ax.scatter(
            [x_mid], [q_tube], marker="s", s=82, color=color,
            edgecolors="white", linewidths=0.8, alpha=alpha, zorder=10,
        )

        # Node quantities/locations: selected node filled, opposite node open.
        ax.scatter(
            [selected_x], [selected_y], marker="o", s=62, color=color,
            edgecolors="white", linewidths=0.8, alpha=alpha, zorder=10,
        )
        ax.scatter(
            [other_x], [other_y], marker="o", s=66, facecolors="white",
            edgecolors=color, linewidths=2.0, alpha=alpha, zorder=10,
        )

    _, _, _, bnode, enode = selected_tube_metadata(diagnostics, int(selected_tube))
    draw_active_tube(
        int(selected_tube), int(selected_node), int(time_index),
        base_x, base_q,
        highlight_color,
        f"Selected tube {selected_tube} ({bnode}–{enode})",
    )

    if comparison_time_index is not None:
        comp_x, comp_q = _conduit_flow_profile_data(
            diagnostics, int(comparison_time_index)
        )
        comp_time = float(diagnostics["times"][comparison_time_index])
        ax.plot(
            comp_x, comp_q, linewidth=2.2, marker="s", markersize=3.4,
            linestyle="--", color=conduit_color, alpha=0.72,
            label=f"Boundary + tube flow — {format_elapsed_time(comp_time)} (comparison)",
        )
        comp_tube = int(comparison_tube if comparison_tube is not None else selected_tube)
        comp_node = int(comparison_node if comparison_node is not None else selected_node)
        _, _, _, cb, ce = selected_tube_metadata(diagnostics, comp_tube)
        draw_active_tube(
            comp_tube, comp_node, int(comparison_time_index),
            comp_x, comp_q,
            COMPARISON_REFERENCE_COLOR,
            f"Comparison tube {comp_tube} ({cb}–{ce})",
            linestyle="--", alpha=0.9,
        )

    # Endpoint values remain explicit external boundary fluxes, not tube Q.
    node_x = np.asarray(diagnostics["node_x"], dtype=float)
    distance = node_x - node_x[0]
    ax.scatter(
        [float(distance[0])],
        [float(diagnostics["spring_outflow"][time_index])],
        marker="s", s=62, color=conduit_color, edgecolors="white",
        linewidths=0.7, zorder=8, label="Spring outflow (−QFIX)",
    )
    ax.scatter(
        [float(distance[-1])],
        [float(diagnostics["direct_recharge_total"][time_index])],
        marker="s", s=62, color=conduit_color, edgecolors="white",
        linewidths=0.7, zorder=8, label="Sinkhole inflow (DIRECT RECHARGE)",
    )

    ax.set_title("Conduit flow along the conduit")
    ax.set_xlabel("Distance along conduit from spring/outlet [m]")
    ax.set_ylabel("Flow [m³/s]")
    ax.set_ylim(0.0, float(flow_max))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_exchange_flow_profile_plot_v7(
    diagnostics: dict,
    time_index: int,
    conduit_color: str,
    selected_node: int,
    selected_tube: int,
    exchange_limit: float,
    comparison_time_index: int | None = None,
    comparison_node: int | None = None,
    comparison_tube: int | None = None,
):
    """Plot signed CFP matrix–conduit exchange at conduit nodes."""
    node_x = np.asarray(diagnostics["node_x"], dtype=float)
    distance = node_x - node_x[0]
    base = np.asarray(diagnostics["exchange_flow"][time_index, :], dtype=float)
    base_time = float(diagnostics["times"][time_index])
    highlight_color = complementary_color(conduit_color)

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.plot(
        distance, base, linewidth=2.0, marker="o", markersize=3.8,
        color=MATRIX_HEAD_COLOR,
        label=f"Exchange — {format_elapsed_time(base_time)}",
    )
    ax.axhline(0.0, linewidth=1.0, color=REFERENCE_COLOR, alpha=0.7)

    def draw_active_tube(
        tube_number: int,
        node_number: int,
        values: np.ndarray,
        color: str,
        label: str,
        linestyle: str = "-",
        alpha: float = 1.0,
    ) -> None:
        _, begin_idx, end_idx, begin_node, end_node = selected_tube_metadata(
            diagnostics, int(tube_number)
        )
        x1 = float(distance[begin_idx])
        x2 = float(distance[end_idx])
        y1 = float(values[begin_idx])
        y2 = float(values[end_idx])
        if int(node_number) == begin_node:
            sx, sy, ox, oy = x1, y1, x2, y2
        elif int(node_number) == end_node:
            sx, sy, ox, oy = x2, y2, x1, y1
        else:
            raise ValueError(
                f"Selected tube {tube_number} is not adjacent to node {node_number}."
            )
        ax.plot(
            [x1, x2], [y1, y2], color=color, linewidth=3.0,
            linestyle=linestyle, alpha=alpha, label=label, zorder=8,
        )
        ax.scatter(
            [sx], [sy], marker="o", s=62, color=color,
            edgecolors="white", linewidths=0.8, alpha=alpha, zorder=10,
        )
        ax.scatter(
            [ox], [oy], marker="o", s=64, facecolors="white",
            edgecolors=color, linewidths=2.0, alpha=alpha, zorder=10,
        )

    _, _, _, bnode, enode = selected_tube_metadata(diagnostics, int(selected_tube))
    draw_active_tube(
        int(selected_tube), int(selected_node), base, highlight_color,
        f"Selected tube {selected_tube} ({bnode}–{enode})",
    )

    if comparison_time_index is not None:
        comp = np.asarray(
            diagnostics["exchange_flow"][comparison_time_index, :], dtype=float
        )
        comp_time = float(diagnostics["times"][comparison_time_index])
        ax.plot(
            distance, comp, linewidth=2.0, marker="o", markersize=3.6,
            linestyle="--", color=MATRIX_HEAD_COLOR, alpha=0.62,
            label=f"Exchange — {format_elapsed_time(comp_time)} (comparison)",
        )
        comp_tube = int(comparison_tube if comparison_tube is not None else selected_tube)
        comp_node = int(comparison_node if comparison_node is not None else selected_node)
        _, _, _, cb, ce = selected_tube_metadata(diagnostics, comp_tube)
        draw_active_tube(
            comp_tube, comp_node, comp, COMPARISON_REFERENCE_COLOR,
            f"Comparison tube {comp_tube} ({cb}–{ce})",
            linestyle="--", alpha=0.9,
        )

    ax.set_title("Matrix–conduit exchange along the conduit")
    ax.set_xlabel("Distance along conduit from spring/outlet [m]")
    ax.set_ylabel("Exchange flow [m³/s]")
    ax.set_ylim(-float(exchange_limit), float(exchange_limit))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_tube_flow_timeseries_plot_v7(
    diagnostics: dict,
    tube_number: int,
    conduit_color: str,
    selected_time_index: int,
    flow_max: float,
    comparison_tube: int | None = None,
    comparison_time_index: int | None = None,
):
    """Plot actual tube Q magnitude through time for one or two selected tubes."""
    times_h = np.asarray(diagnostics["times"], dtype=float) / 3600.0
    tube_idx, _, _, begin_node, end_node = selected_tube_metadata(
        diagnostics, int(tube_number)
    )
    base = np.abs(np.asarray(diagnostics["tube_flow"][:, tube_idx], dtype=float))
    ring_color = complementary_color(conduit_color)

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.plot(
        times_h, base, linewidth=2.2, color=conduit_color,
        label=f"Tube {tube_number} ({begin_node}–{end_node})",
    )
    base_t = float(times_h[selected_time_index])
    ax.axvline(base_t, linestyle=":", linewidth=1.4, color=ring_color, alpha=0.9)
    ax.scatter(
        [base_t], [float(base[selected_time_index])], s=145,
        facecolors="none", edgecolors=ring_color, linewidths=2.4,
        label="Base diagnostic time", zorder=8,
    )

    if comparison_tube is not None:
        comp_idx, _, _, cb, ce = selected_tube_metadata(
            diagnostics, int(comparison_tube)
        )
        comp = np.abs(
            np.asarray(diagnostics["tube_flow"][:, comp_idx], dtype=float)
        )
        ax.plot(
            times_h, comp, linewidth=2.2, linestyle="--",
            color=conduit_color, alpha=0.72,
            label=f"Tube {comparison_tube} ({cb}–{ce}) — comparison",
        )
        if comparison_time_index is not None:
            comp_t = float(times_h[comparison_time_index])
            ax.axvline(
                comp_t, linestyle="--", linewidth=1.2,
                color=COMPARISON_REFERENCE_COLOR, alpha=0.8,
            )
            ax.scatter(
                [comp_t], [float(comp[comparison_time_index])], s=125,
                marker="D", facecolors="none",
                edgecolors=COMPARISON_REFERENCE_COLOR, linewidths=2.2,
                label="Comparison diagnostic time", zorder=8,
            )

    ax.set_title("Transient conduit flow in selected tube(s)")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Tube flow magnitude [m³/s]")
    ax.set_ylim(0.0, float(flow_max))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_node_exchange_flow_timeseries_plot_v6(
    diagnostics: dict,
    node_number: int,
    conduit_color: str,
    selected_time_index: int,
    exchange_limit: float,
    comparison_node: int | None = None,
    comparison_time_index: int | None = None,
):
    """Plot transient signed exchange at one or two selected conduit nodes."""
    times_h = np.asarray(diagnostics["times"], dtype=float) / 3600.0
    base_idx, _ = selected_node_metadata(diagnostics, node_number)
    base = np.asarray(diagnostics["exchange_flow"][:, base_idx], dtype=float)
    ring_color = complementary_color(conduit_color)

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.plot(times_h, base, linewidth=2.0, color=MATRIX_HEAD_COLOR,
            label=f"Exchange — node {node_number}")
    ax.axhline(0.0, linewidth=1.0, color=REFERENCE_COLOR, alpha=0.7)
    base_t = float(times_h[selected_time_index])
    ax.axvline(base_t, linestyle=":", linewidth=1.4, color=ring_color, alpha=0.9)
    ax.scatter([base_t], [float(base[selected_time_index])], s=145, facecolors="none",
               edgecolors=ring_color, linewidths=2.4, label="Base selection", zorder=8)

    if comparison_node is not None:
        compare_idx, _ = selected_node_metadata(diagnostics, int(comparison_node))
        compare = np.asarray(diagnostics["exchange_flow"][:, compare_idx], dtype=float)
        ax.plot(times_h, compare, linewidth=2.0, linestyle="--", color=MATRIX_HEAD_COLOR,
                alpha=0.62, label=f"Exchange — node {comparison_node} (comparison)")
        if comparison_time_index is not None:
            compare_t = float(times_h[comparison_time_index])
            ax.axvline(compare_t, linestyle="--", linewidth=1.2,
                       color=COMPARISON_REFERENCE_COLOR, alpha=0.8)
            ax.scatter([compare_t], [float(compare[comparison_time_index])], s=125,
                       marker="D", facecolors="none", edgecolors=COMPARISON_REFERENCE_COLOR,
                       linewidths=2.2, label="Comparison selection", zorder=8)

    ax.set_title("Transient matrix–conduit exchange at selected node(s)")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Exchange flow [m³/s]")
    ax.set_ylim(-float(exchange_limit), float(exchange_limit))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_flow_profile_comparison_plots_v7(
    scenarios: list[dict],
    target_time: float,
    node_number: int,
    tube_number: int,
):
    """Compare actual conduit-flow profiles and exchange profiles across runs."""
    labels = _scenario_labels(scenarios)
    fig_q, ax_q = plt.subplots(figsize=(9.5, 5.0))
    fig_ex, ax_ex = plt.subplots(figsize=(9.5, 5.0))

    for scenario in scenarios:
        diagnostics = scenario["diagnostics"]
        idx = _nearest_diagnostic_time_index(diagnostics, target_time)
        x_q, q = _conduit_flow_profile_data(diagnostics, idx)
        node_distance = (
            np.asarray(diagnostics["node_x"], dtype=float)
            - float(diagnostics["node_x"][0])
        )
        ex = np.asarray(diagnostics["exchange_flow"][idx, :], dtype=float)
        label = labels[int(scenario["execution_number"])]
        ax_q.plot(
            x_q, q, linewidth=2.0, marker="s", markersize=3.0,
            color=scenario["color"], label=label
        )
        ax_ex.plot(
            node_distance, ex, linewidth=2.0, marker="o", markersize=3.0,
            color=scenario["color"], label=label
        )

    first_d = scenarios[0]["diagnostics"]
    node_idx, _ = selected_node_metadata(first_d, int(node_number))
    node_distance = (
        np.asarray(first_d["node_x"], dtype=float) - float(first_d["node_x"][0])
    )
    ax_ex.axvline(
        float(node_distance[node_idx]), linestyle=":", linewidth=1.3,
        color=REFERENCE_COLOR,
    )
    x1, x2, b, e = _tube_profile_geometry(first_d, int(tube_number))
    for ax in (ax_q, ax_ex):
        ax.axvspan(
            min(x1, x2), max(x1, x2),
            facecolor="none", edgecolor=REFERENCE_COLOR,
            hatch="//", linewidth=0.0, alpha=0.25,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax_q.set_title(
        f"Stored runs — conduit flow at t = {format_elapsed_time(target_time)} "
        f"(selected tube {tube_number}: {b}–{e})"
    )
    ax_q.set_xlabel("Distance along conduit from spring/outlet [m]")
    ax_q.set_ylabel("Flow [m³/s]")
    ax_q.set_ylim(bottom=0.0)
    fig_q.tight_layout()

    ax_ex.axhline(0.0, linewidth=1.0, color=REFERENCE_COLOR, alpha=0.7)
    ax_ex.set_title(
        f"Stored runs — matrix–conduit exchange at t = "
        f"{format_elapsed_time(target_time)}"
    )
    ax_ex.set_xlabel("Distance along conduit from spring/outlet [m]")
    ax_ex.set_ylabel("Exchange flow [m³/s]")
    fig_ex.tight_layout()
    return fig_q, fig_ex


def make_flow_timeseries_comparison_plots_v7(
    scenarios: list[dict],
    node_number: int,
    tube_number: int,
):
    """Compare actual selected-tube flow and selected-node exchange across runs."""
    labels = _scenario_labels(scenarios)
    fig_q, ax_q = plt.subplots(figsize=(9.5, 5.0))
    fig_ex, ax_ex = plt.subplots(figsize=(9.5, 5.0))

    for scenario in scenarios:
        diagnostics = scenario["diagnostics"]
        tube_idx, _, _, _, _ = selected_tube_metadata(
            diagnostics, int(tube_number)
        )
        node_idx, _ = selected_node_metadata(diagnostics, int(node_number))
        times_h = np.asarray(diagnostics["times"], dtype=float) / 3600.0
        q = np.abs(np.asarray(diagnostics["tube_flow"][:, tube_idx], dtype=float))
        ex = np.asarray(diagnostics["exchange_flow"][:, node_idx], dtype=float)
        label = labels[int(scenario["execution_number"])]
        ax_q.plot(times_h, q, linewidth=2.1, color=scenario["color"], label=label)
        ax_ex.plot(times_h, ex, linewidth=2.1, color=scenario["color"], label=label)

    ax_q.set_title(f"Stored runs — transient flow in tube {tube_number}")
    ax_q.set_xlabel("Time [h]")
    ax_q.set_ylabel("Tube flow magnitude [m³/s]")
    ax_q.set_ylim(bottom=0.0)
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=8)
    fig_q.tight_layout()

    ax_ex.axhline(0.0, linewidth=1.0, color=REFERENCE_COLOR, alpha=0.7)
    ax_ex.set_title(
        f"Stored runs — transient matrix–conduit exchange at node {node_number}"
    )
    ax_ex.set_xlabel("Time [h]")
    ax_ex.set_ylabel("Exchange flow [m³/s]")
    ax_ex.grid(True, alpha=0.3)
    ax_ex.legend(fontsize=8)
    fig_ex.tight_layout()
    return fig_q, fig_ex


# -----------------------------------------------------------------------------
# 4.5 Cumulative water-budget plots
# -----------------------------------------------------------------------------
def _budget_plot_data(budget: dict) -> tuple[list[str], np.ndarray]:
    """Return requested whole-system budget categories using a source/sink sign convention."""
    labels = [
        "Diffuse\nrecharge",
        "Direct\nrecharge",
        "Matrix storage\nchange",
        "Matrix boundary\noutflow",
        "Karst conduit\noutflow",
    ]
    # Positive = source to the combined matrix/conduit system. A positive matrix
    # storage change means water was accumulated in matrix storage and is therefore
    # shown as a sink (negative plotted value). Storage release appears positive.
    values = np.asarray(
        [
            float(budget["diffuse_recharge"]),
            float(budget["direct_recharge"]),
            -float(budget["matrix_storage_change"]),
            -float(budget["matrix_boundary_outflow"]),
            -float(budget["karst_conduit_outflow"]),
        ],
        dtype=float,
    )
    return labels, values


def make_cumulative_budget_plot(run: dict):
    """Plot the requested cumulative whole-run budget for one model execution."""
    budget = run.get("budget")
    if budget is None:
        raise ValueError("No cumulative budget is available for this run.")
    labels, values = _budget_plot_data(budget)

    fig, ax = plt.subplots(figsize=(9.5, 5.1))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.68, color=run["color"], alpha=0.88)
    ax.axhline(0.0, linewidth=1.0, color=REFERENCE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cumulative water volume [m³]")
    ax.set_title("Cumulative water budget — complete simulation")
    ax.grid(True, axis="y", alpha=0.25)

    finite = values[np.isfinite(values)]
    if finite.size:
        span = max(float(np.max(np.abs(finite))), 1.0)
        offset = 0.018 * span
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0.0 else "top"
            y = float(value + offset if value >= 0.0 else value - offset)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                f"{value:,.1f}",
                ha="center",
                va=va,
                fontsize=8.5,
            )
        ax.margins(y=0.14)

    fig.tight_layout()
    return fig


def make_budget_comparison_plot(scenarios: list[dict]):
    """Compare cumulative whole-run budgets while retaining each run's color."""
    if not scenarios:
        raise ValueError("No runs were supplied for budget comparison.")

    labels, _ = _budget_plot_data(scenarios[0]["budget"])
    x = np.arange(len(labels), dtype=float)
    n = len(scenarios)
    width = min(0.16, 0.78 / max(n, 1))
    offsets = (np.arange(n) - 0.5 * (n - 1)) * width

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for offset, scenario in zip(offsets, scenarios):
        _, values = _budget_plot_data(scenario["budget"])
        ax.bar(
            x + offset,
            values,
            width=width,
            color=scenario["color"],
            alpha=0.88,
            label=scenario["name"],
        )

    ax.axhline(0.0, linewidth=1.0, color=REFERENCE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cumulative water volume [m³]")
    ax.set_title("Cumulative water-budget comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


# =============================================================================
# 5. STREAMLIT USER INTERFACE
# =============================================================================
# -----------------------------------------------------------------------------
# 5.1 Session-state initialization and migration
# -----------------------------------------------------------------------------
# The stored-run data schema is unchanged from v7. This revision changes only
# diagnostic widget selection and plotting, so existing v7 runs remain usable.
if st.session_state.get("app_state_schema_version") != APP_STATE_SCHEMA_VERSION:
    st.session_state.saved_scenarios = []
    st.session_state.current_run = None
    st.session_state.total_run_count = 0
    for key in list(st.session_state.keys()):
        if key.startswith(("comparison_", "diagnostic_", "show_heads_", "show_flows_", "show_budget_")):
            del st.session_state[key]
    st.session_state.app_state_schema_version = APP_STATE_SCHEMA_VERSION

if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []

if "current_run" not in st.session_state:
    st.session_state.current_run = None

if "total_run_count" not in st.session_state:
    st.session_state.total_run_count = 0

# Development-session migration: older versions of this app stored manually
# named scenarios without an execution_number. Streamlit can preserve session
# state across a source-code reload, so reset only those incompatible in-memory
# objects rather than failing later with a KeyError.
legacy_history = any(
    any(key not in item for key in ("execution_number", "slot_number", "color"))
    or int(item.get("slot_number", MAX_STORED_RUNS + 1)) > MAX_STORED_RUNS
    for item in st.session_state.saved_scenarios
)
legacy_current = (
    st.session_state.current_run is not None
    and any(
        key not in st.session_state.current_run
        for key in ("execution_number", "slot_number", "color")
    )
)
if legacy_history or legacy_current:
    st.session_state.saved_scenarios = []
    st.session_state.current_run = None
    st.session_state.total_run_count = 0
    if "comparison_run_selection" in st.session_state:
        del st.session_state.comparison_run_selection


# =============================================================================
# User interface
# =============================================================================


# -----------------------------------------------------------------------------
# 5.2 Model setup, parameter inputs, and model execution
# -----------------------------------------------------------------------------
st.title("💧 Karst Spring Response")
st.markdown(
    "Explore how conduit and matrix properties influence the simulated spring "
    "response. Every successful execution is stored automatically for comparison; "
    "only the last 5 executions are retained."
)

st.header("1. Model setup and run")
st.caption(
    "Changing a model control updates the setup but does **not** execute CFP. "
    "If the controls no longer match the last execution, the current-result "
    "display disappears; that completed run remains safely stored in the "
    "rolling comparison history."
)

# One central input-mode toggle controls all numerical model inputs. The mode is
# intentionally not part of the model parameter fingerprint: switching between
# slider and number input must preserve the represented values and the current run.
INPUT_MODE_KEY = "model_use_number_inputs"
use_number_inputs = st.toggle(
    "Use number inputs",
    value=False,
    key=INPUT_MODE_KEY,
    help=(
        "Switch all numerical model controls between sliders and direct number "
        "inputs. Existing values are preserved when the input mode changes."
    ),
)
st.caption(
    "Input mode: **number inputs**" if use_number_inputs else "Input mode: **sliders**"
)

# Relative roughness k/D is the canonical conduit-roughness value.  Absolute
# roughness k is only an alternative UI representation.  Therefore changing
# conduit diameter preserves k/D and updates k = (k/D) * D.
# The relative-roughness range is fixed and independent of conduit diameter;
# the available absolute-roughness range scales with the current diameter.
ROUGHNESS_ABS_STATE_KEY = "roughness_absolute__value"
ROUGHNESS_REL_STATE_KEY = "roughness_relative__value"
ROUGHNESS_MODE_KEY = "conduit_use_absolute_roughness"
ROUGHNESS_PREVIOUS_MODE_KEY = "conduit_use_absolute_roughness_previous_mode"
ROUGHNESS_PREVIOUS_DIAMETER_KEY = "conduit_roughness_previous_diameter"
ROUGHNESS_RELATIVE_MIN = 1.0e-7
ROUGHNESS_RELATIVE_MAX = 1.25

# Fresh-session defaults: d = 0.30 m, k/D = 0.10, hence k = 0.03 m.
# Relative roughness is the initial UI representation (toggle off).
st.session_state.setdefault(ROUGHNESS_MODE_KEY, False)
st.session_state.setdefault(ROUGHNESS_REL_STATE_KEY, 0.10)
st.session_state.setdefault(ROUGHNESS_ABS_STATE_KEY, 0.03)
st.session_state.setdefault(ROUGHNESS_PREVIOUS_MODE_KEY, False)
st.session_state.setdefault(ROUGHNESS_PREVIOUS_DIAMETER_KEY, 0.30)

with st.expander("Conduit properties", expanded=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        dmt = synced_numeric_input(
            "Conduit diameter, d [m]",
            base_key="conduit_diameter",
            minimum=0.02,
            maximum=1.00,
            default=0.30,
            step=0.01,
            number_mode=use_number_inputs,
            format_string="%.2f",
        )
        trtst = synced_numeric_input(
            "Tortuosity [-]",
            base_key="conduit_tortuosity",
            minimum=1.0,
            maximum=3.0,
            default=1.0,
            step=0.05,
            number_mode=use_number_inputs,
            format_string="%.2f",
        )

    with c2:
        st.markdown("Roughness input: rel. k/d or absolute k")
        use_absolute_roughness = st.toggle(
            "Use absolute k",
            key=ROUGHNESS_MODE_KEY,
            help=(
                "Off: enter relative roughness k/d. On: enter absolute roughness "
                "height k [m]. Relative roughness is the underlying value, so "
                "changing conduit diameter preserves k/D and adjusts k."
            ),
        )

        previous_roughness_mode = bool(
            st.session_state.get(ROUGHNESS_PREVIOUS_MODE_KEY, False)
        )
        previous_roughness_diameter = float(
            st.session_state.get(ROUGHNESS_PREVIOUS_DIAMETER_KEY, 0.30)
        )
        roughness_mode_changed = (
            bool(use_absolute_roughness) != previous_roughness_mode
        )
        roughness_diameter_changed = not np.isclose(
            float(dmt),
            previous_roughness_diameter,
            rtol=0.0,
            atol=1.0e-12,
        )

        # k/D is the canonical roughness state.  Clamp any legacy session value
        # once to the fixed relative-roughness range.
        roughness_rel = _clip_numeric(
            st.session_state.get(ROUGHNESS_REL_STATE_KEY, 0.10),
            ROUGHNESS_RELATIVE_MIN,
            ROUGHNESS_RELATIVE_MAX,
        )
        st.session_state[ROUGHNESS_REL_STATE_KEY] = roughness_rel

        if use_absolute_roughness:
            # Absolute roughness is a diameter-dependent view of canonical k/D.
            # A diameter or representation change refreshes the widget from k/D;
            # a direct user edit of k is converted back to the canonical k/D.
            absolute_min = ROUGHNESS_RELATIVE_MIN * float(dmt)
            absolute_max = ROUGHNESS_RELATIVE_MAX * float(dmt)

            if roughness_mode_changed or roughness_diameter_changed:
                st.session_state[ROUGHNESS_ABS_STATE_KEY] = (
                    roughness_rel * float(dmt)
                )

            rh = synced_numeric_input(
                "Absolute roughness, k [m]",
                base_key="roughness_absolute",
                minimum=absolute_min,
                maximum=absolute_max,
                default=0.10 * float(dmt),
                step=ROUGHNESS_RELATIVE_MIN * float(dmt),
                number_mode=use_number_inputs,
                format_string="%.7g",
                force_sync=(roughness_mode_changed or roughness_diameter_changed),
            )
            roughness_rel = _clip_numeric(
                float(rh) / max(float(dmt), 1.0e-12),
                ROUGHNESS_RELATIVE_MIN,
                ROUGHNESS_RELATIVE_MAX,
            )
            rh = roughness_rel * float(dmt)
            st.session_state[ROUGHNESS_REL_STATE_KEY] = roughness_rel
            st.session_state[ROUGHNESS_ABS_STATE_KEY] = rh
            st.caption(f"Equivalent relative roughness: k/D = {roughness_rel:.4g}")

        else:
            # Relative mode displays the canonical value directly.  Its slider
            # range is fixed, and a diameter change deliberately does not refresh
            # or move this widget.  Only the derived absolute roughness changes.
            relative_roughness = synced_numeric_input(
                "Relative roughness, k/D [-]",
                base_key="roughness_relative",
                minimum=ROUGHNESS_RELATIVE_MIN,
                maximum=ROUGHNESS_RELATIVE_MAX,
                default=0.10,
                step=ROUGHNESS_RELATIVE_MIN,
                number_mode=use_number_inputs,
                format_string="%.4g",
                force_sync=roughness_mode_changed,
            )
            roughness_rel = _clip_numeric(
                float(relative_roughness),
                ROUGHNESS_RELATIVE_MIN,
                ROUGHNESS_RELATIVE_MAX,
            )
            rh = roughness_rel * float(dmt)
            st.session_state[ROUGHNESS_REL_STATE_KEY] = roughness_rel
            st.session_state[ROUGHNESS_ABS_STATE_KEY] = rh
            st.caption(f"Equivalent roughness height: k = {rh:.7g} m")

        # These are UI-state trackers only; neither is a model parameter.
        st.session_state[ROUGHNESS_PREVIOUS_MODE_KEY] = bool(use_absolute_roughness)
        st.session_state[ROUGHNESS_PREVIOUS_DIAMETER_KEY] = float(dmt)

        cfptemp = synced_numeric_input(
            "Water temperature [°C]",
            base_key="water_temperature",
            minimum=0.0,
            maximum=30.0,
            default=10.0,
            step=1.0,
            number_mode=use_number_inputs,
            format_string="%.0f",
        )

    with c3:
        laminar_only = st.toggle(
            "Laminar only",
            value=False,
            help=(
                "When enabled, the critical Reynolds-number controls are locked and "
                "CFP receives both transition thresholds multiplied by 10,000. This "
                "keeps the modeled conduit in the laminar regime over a much wider "
                "range of simulated velocities."
            ),
        )
        lcrey = synced_numeric_input(
            "Lower critical Reynolds number [-]",
            base_key="lower_critical_reynolds",
            minimum=100,
            maximum=3000,
            default=500,
            step=50,
            number_mode=use_number_inputs,
            disabled=laminar_only,
            integer=True,
        )
        hcrey = synced_numeric_input(
            "Higher critical Reynolds number [-]",
            base_key="higher_critical_reynolds",
            minimum=1000,
            maximum=10000,
            default=5000,
            step=100,
            number_mode=use_number_inputs,
            disabled=laminar_only,
            integer=True,
        )
        if laminar_only:
            st.caption(
                f"Effective CFP thresholds: Re₁ = {int(lcrey * 10000):,}, "
                f"Re₂ = {int(hcrey * 10000):,}."
            )

with st.expander("Matrix, exchange and storage", expanded=True):
    c4, c5, c6 = st.columns(3)

    with c4:
        kxch = parameter_input(
            "Conduit wall permeability [m/s]",
            key="conduit_wall_permeability",
            default=4.0e-5,
            min_value=1.0e-8,
            max_value=1.0e-2,
            scale="log",
            use_number_input=use_number_inputs,
            number_format="%.2e",
            log_steps_per_decade=20,
        )
        st.caption(f"kₓch = {kxch:.2e} m/s")

    with c5:
        hk = parameter_input(
            "Matrix hydraulic conductivity [m/s]",
            key="matrix_hydraulic_conductivity",
            default=1.0e-4,
            min_value=1.0e-8,
            max_value=1.0e-2,
            scale="log",
            use_number_input=use_number_inputs,
            number_format="%.2e",
            log_steps_per_decade=20,
        )
        st.caption(f"K = {hk:.2e} m/s")

    with c6:
        sy = synced_numeric_input(
            "Matrix specific yield, Sy [-]",
            base_key="matrix_specific_yield",
            minimum=0.001,
            maximum=0.40,
            default=0.05,
            step=0.001,
            number_mode=use_number_inputs,
            format_string="%.3f",
        )
        CADS = synced_numeric_input(
            "Conduit associated storage (CADS)",
            base_key="conduit_associated_storage",
            minimum=0.0,
            maximum=1.0,
            default=0.0,
            step=0.01,
            number_mode=use_number_inputs,
            format_string="%.2f",
        )

params = {
    "dmt": dmt,
    "trtst": trtst,
    "rh": rh,
    "lcrey": lcrey,
    "hcrey": hcrey,
    "kxch": kxch,
    "cfptemp": cfptemp,
    "hk": hk,
    "sy": sy,
    "CADS": CADS,
    "laminar_only": laminar_only,
}

# Persist a user-edited current-run name before a parameter change can hide the
# current-result section or a new execution can replace current_run. This also
# catches a text edit when the user clicks directly into another control.
sync_current_run_name_from_widget()

# A control change reruns only the Streamlit script, not the numerical model.
# Hide the current result immediately once the displayed result no longer
# corresponds to the visible parameter controls. The result itself has already
# been stored automatically in saved_scenarios after a successful execution.
if (
    st.session_state.current_run is not None
    and not parameter_sets_equal(st.session_state.current_run["params"], params)
):
    st.session_state.current_run = None

run_model = st.button(
    "▶️ Run model",
    type="primary",
    use_container_width=True,
)

if run_model:
    if hcrey <= lcrey:
        st.error(
            "The higher critical Reynolds number must be larger than the lower "
            "critical Reynolds number."
        )
    else:
        try:
            with st.spinner("Running CFP/MODFLOW model..."):
                model_result = cfpy_model(**params)

            new_run = {
                "params": params.copy(),
                **model_result,
            }
            stored_run = store_run_in_rolling_history(new_run)
            # The current-result view references the same stored object, avoiding
            # a second in-memory copy of the full matrix-head time series.
            st.session_state.current_run = stored_run

            retained = len(st.session_state.saved_scenarios)
            st.success(
                f"Model run completed and stored automatically as "
                f"**{stored_run['name']}** ({retained}/{MAX_STORED_RUNS} stored runs)."
            )
        except Exception as exc:
            st.session_state.current_run = None
            st.error(f"Model run failed: {exc}")


# =============================================================================
# -----------------------------------------------------------------------------
# 5.3 Current result and optional diagnostics
# -----------------------------------------------------------------------------
# =============================================================================
if st.session_state.current_run is not None:
    current = st.session_state.current_run

    st.markdown("#### Current result")
    st.caption(
        "The run is already stored automatically. You can rename it here; the "
        "edited name remains attached to this run when you change parameters or "
        "execute the model again."
    )

    run_name_key = f"run_name_input_{int(current['execution_number'])}"
    if run_name_key not in st.session_state:
        st.session_state[run_name_key] = current["name"]
    run_name = st.text_input(
        "Run name",
        key=run_name_key,
        help=(
            "Default names are run1 ... run5. You may replace the default with "
            "a descriptive name before continuing with another parameter set."
        ),
    )
    cleaned_run_name = run_name.strip() or f"run{int(current['slot_number'])}"
    if cleaned_run_name != current["name"]:
        current["name"] = cleaned_run_name
        for item in st.session_state.saved_scenarios:
            if int(item["execution_number"]) == int(current["execution_number"]):
                item["name"] = cleaned_run_name
                break

    fig_current, ax_current = plt.subplots(figsize=(9.5, 4.8))
    ax_current.plot(
        current["times"] / 3600.0,
        current["spring_flow"],
        linewidth=2.3,
        color=current["color"],
        label=(
            "Spring outflow (−QFIX"
            + (
                f", node {current['spring_node_number']}"
                if current.get("spring_node_number") is not None
                else ""
            )
            + ")"
        ),
    )
    ax_current.plot(
        current["times"] / 3600.0,
        current.get("direct_recharge_flow", current["inlet_flow"]),
        linewidth=2.0,
        linestyle="--",
        color=current["color"],
        alpha=0.72,
        label=(
            "Sinkhole inflow (DIRECT RECHARGE"
            + (
                f", node {current['inlet_node_number']}"
                if current.get("inlet_node_number") is not None
                else ""
            )
            + ")"
        ),
    )
    ax_current.set_xlabel("Time [h]")
    ax_current.set_ylabel("External conduit flow [m³/s]")
    ax_current.grid(True, alpha=0.3)
    ax_current.legend()
    fig_current.tight_layout()
    st.pyplot(fig_current, use_container_width=True)
    plt.close(fig_current)
    st.caption(
        "These two curves are external CFP boundary fluxes from the node table: "
        "sinkhole inflow is DIRECT RECHARGE, while spring outflow is the magnitude "
        "of negative QFIX. They intentionally differ from the adjacent tube Q "
        "because matrix–conduit exchange can occur at the boundary nodes."
    )

    # -------------------------------------------------------------------------
    # Optional budget, head and flow diagnostics
    # -------------------------------------------------------------------------
    diagnostic_toggle_a, diagnostic_toggle_b, diagnostic_toggle_c = st.columns(3)
    with diagnostic_toggle_a:
        show_head_diagnostics = st.toggle(
            "Show head diagnostics",
            value=False,
            help=(
                "Explore the matrix head field in plan view, conduit/matrix profiles, "
                "and transient heads at a selected conduit node."
            ),
            key=f"show_heads_{current['execution_number']}",
        )
    with diagnostic_toggle_b:
        show_flow_diagnostics = st.toggle(
            "Show flow diagnostics",
            value=False,
            help=(
                "Explore conduit flow and matrix–conduit exchange along the conduit "
                "and through time at a selected conduit node."
            ),
            key=f"show_flows_{current['execution_number']}",
        )
    with diagnostic_toggle_c:
        show_budget = st.toggle(
            "Show cumulative budget",
            value=False,
            help=(
                "Show cumulative external recharge, storage change and outlet "
                "components over the complete simulation."
            ),
            key=f"show_budget_{current['execution_number']}",
        )

    if show_budget:
        budget = current.get("budget")
        if budget is None:
            st.warning(
                "The cumulative budget is unavailable: "
                + (current.get("budget_error") or "unknown listing-file parsing error.")
            )
        else:
            with st.expander("Cumulative water budget — complete run", expanded=True):
                fig_budget = make_cumulative_budget_plot(current)
                st.pyplot(fig_budget, use_container_width=True)
                plt.close(fig_budget)
                st.caption(
                    "Sign convention: recharge and storage release are positive; "
                    "matrix storage gain and external outflows are negative. "
                    f"Budget residual = {budget['residual']:.3g} m³."
                )

    if show_head_diagnostics or show_flow_diagnostics:
        diagnostics = current.get("diagnostics")

        if diagnostics is None:
            message = current.get("diagnostics_error") or "Unknown post-processing error."
            st.warning(
                "The spring-response simulation succeeded, but the requested "
                f"diagnostics are unavailable: {message}"
            )
        else:
            n_profile_times = len(diagnostics["times"])
            default_time_idx = int(
                np.argmin(np.abs(diagnostics["times"] - (1.0 + 2.0 * 3600.0)))
            )
            run_key = int(current["execution_number"])
            conduit_color = current["color"]
            node_options = np.asarray(diagnostics["node_numbers"], dtype=int).tolist()

            # -----------------------------------------------------------------
            # Shared controls for both diagnostic families
            # -----------------------------------------------------------------
            st.markdown("##### Diagnostic controls")
            st.caption(
                "The base time and conduit node are shared by all visible diagnostics. "
                "After choosing the node, select its left or right adjacent tube. "
                "Head/exchange quantities are node based; conduit-flow quantities use "
                "the actual selected tube Q. Optional comparison mode provides the same "
                "node-centered selection for a second diagnostic state."
            )

            shared_time_key = f"diagnostic_shared_time_{run_key}"
            shared_node_key = f"diagnostic_shared_node_{run_key}"
            shared_tube_side_key = f"diagnostic_shared_tube_side_{run_key}"

            if shared_time_key not in st.session_state:
                old_key = f"head_shared_time_{run_key}"
                st.session_state[shared_time_key] = int(
                    st.session_state.get(old_key, default_time_idx)
                )
            if (
                shared_node_key not in st.session_state
                or st.session_state[shared_node_key] not in node_options
            ):
                old_key = f"head_shared_node_{run_key}"
                old_value = st.session_state.get(old_key, node_options[-1])
                st.session_state[shared_node_key] = (
                    int(old_value) if int(old_value) in node_options else node_options[-1]
                )

            selector_a, selector_b, selector_c = st.columns(3)
            with selector_a:
                selected_time_index = st.select_slider(
                    "Diagnostic time — base",
                    options=list(range(n_profile_times)),
                    format_func=lambda idx: format_elapsed_time(
                        float(diagnostics["times"][idx])
                    ),
                    key=shared_time_key,
                )
            with selector_b:
                selected_node = st.select_slider(
                    "Conduit node — base",
                    options=node_options,
                    key=shared_node_key,
                )
            base_tube_choices = adjacent_tubes_for_node(diagnostics, int(selected_node))
            base_sides = list(base_tube_choices)
            if (
                shared_tube_side_key not in st.session_state
                or st.session_state[shared_tube_side_key] not in base_sides
            ):
                st.session_state[shared_tube_side_key] = default_tube_side(
                    diagnostics, int(selected_node)
                )
            with selector_c:
                selected_tube_side = st.radio(
                    "Adjacent tube — base",
                    options=base_sides,
                    horizontal=True,
                    key=shared_tube_side_key,
                    format_func=lambda side: f"{side} (tube {base_tube_choices[side]})",
                    help=(
                        "Choose the tube immediately to the left or right of the "
                        "selected conduit node. End nodes have only one adjacent tube."
                    ),
                )
            selected_tube = int(base_tube_choices[selected_tube_side])

            comparison_mode = st.toggle(
                "Compare a second diagnostic time and conduit node",
                value=False,
                key=f"diagnostic_comparison_mode_{run_key}",
                help=(
                    "When enabled, choose a second time and node plus the left/right "
                    "tube adjacent to that node. Stored-run comparisons use only the "
                    "base selections."
                ),
            )

            comparison_time_index: int | None = None
            comparison_node: int | None = None
            comparison_tube: int | None = None
            comparison_tube_side: str | None = None
            if comparison_mode:
                compare_time_key = f"diagnostic_compare_time_{run_key}"
                compare_node_key = f"diagnostic_compare_node_{run_key}"
                compare_tube_side_key = f"diagnostic_compare_tube_side_{run_key}"

                if compare_time_key not in st.session_state:
                    st.session_state[compare_time_key] = n_profile_times - 1
                if (
                    compare_node_key not in st.session_state
                    or st.session_state[compare_node_key] not in node_options
                ):
                    st.session_state[compare_node_key] = node_options[0]

                compare_a, compare_b, compare_c = st.columns(3)
                with compare_a:
                    comparison_time_index = st.select_slider(
                        "Diagnostic time — comparison",
                        options=list(range(n_profile_times)),
                        format_func=lambda idx: format_elapsed_time(
                            float(diagnostics["times"][idx])
                        ),
                        key=compare_time_key,
                    )
                with compare_b:
                    comparison_node = st.select_slider(
                        "Conduit node — comparison",
                        options=node_options,
                        key=compare_node_key,
                    )
                comparison_tube_choices = adjacent_tubes_for_node(
                    diagnostics, int(comparison_node)
                )
                comparison_sides = list(comparison_tube_choices)
                if (
                    compare_tube_side_key not in st.session_state
                    or st.session_state[compare_tube_side_key] not in comparison_sides
                ):
                    st.session_state[compare_tube_side_key] = default_tube_side(
                        diagnostics, int(comparison_node)
                    )
                with compare_c:
                    comparison_tube_side = st.radio(
                        "Adjacent tube — comparison",
                        options=comparison_sides,
                        horizontal=True,
                        key=compare_tube_side_key,
                        format_func=lambda side: (
                            f"{side} (tube {comparison_tube_choices[side]})"
                        ),
                    )
                comparison_tube = int(
                    comparison_tube_choices[comparison_tube_side]
                )

            selected_node_idx, selected_column = selected_node_metadata(
                diagnostics,
                int(selected_node),
            )
            _, _, _, selected_tube_begin, selected_tube_end = selected_tube_metadata(
                diagnostics,
                int(selected_tube),
            )
            selected_time = float(diagnostics["times"][selected_time_index])
            stress_period = int(diagnostics["stress_periods"][selected_time_index])
            time_step = int(diagnostics["time_steps"][selected_time_index])
            selected_x = float(diagnostics["node_x"][selected_node_idx])
            selected_y = float(diagnostics["node_y"][selected_node_idx])
            selected_row = int(diagnostics["node_rows"][selected_node_idx])

            st.caption(
                f"Base selection: t = {selected_time:.0f} s "
                f"({selected_time / 3600.0:.3f} h), stress period {stress_period}, "
                f"time step {time_step}; conduit node {selected_node} at "
                f"x = {selected_x:.1f} m, y = {selected_y:.1f} m, matrix column "
                f"{selected_column}, row {selected_row}; {selected_tube_side.lower()} "
                f"tube {selected_tube} connects nodes "
                f"{selected_tube_begin}–{selected_tube_end}."
            )

            if (
                comparison_mode
                and comparison_time_index is not None
                and comparison_node is not None
                and comparison_tube is not None
            ):
                _, comp_column = selected_node_metadata(
                    diagnostics, int(comparison_node)
                )
                _, _, _, comp_tube_begin, comp_tube_end = selected_tube_metadata(
                    diagnostics, int(comparison_tube)
                )
                comp_time = float(diagnostics["times"][comparison_time_index])
                comp_sp = int(diagnostics["stress_periods"][comparison_time_index])
                comp_ts = int(diagnostics["time_steps"][comparison_time_index])
                st.caption(
                    f"Comparison selection: t = {comp_time:.0f} s "
                    f"({comp_time / 3600.0:.3f} h), stress period {comp_sp}, "
                    f"time step {comp_ts}; conduit node {comparison_node}, matrix "
                    f"column {comp_column}; {comparison_tube_side.lower()} tube "
                    f"{comparison_tube} connects nodes {comp_tube_begin}–{comp_tube_end}."
                )

            # -----------------------------------------------------------------
            # Diagnostic-specific display limits
            # -----------------------------------------------------------------
            head_ylim: tuple[float, float] | None = None
            matrix_head_max: float | None = None
            flow_max: float | None = None
            exchange_limit: float | None = None

            if show_head_diagnostics:
                matrix_lower, matrix_observed_max = global_matrix_head_range(diagnostics)
                line_lower, _ = global_head_axis_range(diagnostics)

                conduit_values_all = _finite_head_values(diagnostics["conduit_heads"])
                matrix_node_values_all = _finite_head_values(
                    diagnostics["matrix_heads_at_nodes"]
                )
                if conduit_values_all.size == 0:
                    raise ValueError("The conduit-head results contain no valid values.")
                profile_observed_max = float(np.max(conduit_values_all))
                if matrix_node_values_all.size:
                    profile_observed_max = max(
                        profile_observed_max,
                        float(np.max(matrix_node_values_all)),
                    )

                matrix_min_slider, matrix_default_max, matrix_step = (
                    head_ceiling_slider_settings(matrix_lower, matrix_observed_max)
                )
                profile_min_slider, profile_default_max, profile_step = (
                    head_ceiling_slider_settings(line_lower, profile_observed_max)
                )
                matrix_decimals = _head_tick_decimals(
                    np.asarray([0.0, matrix_step], dtype=float)
                )
                profile_decimals = _head_tick_decimals(
                    np.asarray([0.0, profile_step], dtype=float)
                )

                head_limit_a, head_limit_b = st.columns(2)
                with head_limit_a:
                    matrix_head_max = st.slider(
                        "Maximum matrix head in plan view [m]",
                        min_value=float(matrix_min_slider),
                        max_value=float(matrix_default_max),
                        value=float(matrix_default_max),
                        step=float(matrix_step),
                        format=f"%.{matrix_decimals}f",
                        key=f"matrix_head_ceiling_{run_key}",
                        help=(
                            "The default shows the complete matrix-head range. Lower "
                            "the ceiling to emphasize lower-head differences; values "
                            "above it use the highest map color."
                        ),
                    )

                with head_limit_b:
                    profile_head_max = st.slider(
                        "Maximum head in head-profile plots [m]",
                        min_value=float(profile_min_slider),
                        max_value=float(profile_default_max),
                        value=float(profile_default_max),
                        step=float(profile_step),
                        format=f"%.{profile_decimals}f",
                        key=f"profile_head_ceiling_{run_key}",
                        help=(
                            "The default is based on the complete run. Lower it to "
                            "focus on smaller head differences in the spatial and "
                            "transient profile plots."
                        ),
                    )
                head_ylim = (float(line_lower), float(profile_head_max))

            if show_flow_diagnostics:
                observed_flow_max = global_conduit_flow_max(diagnostics)
                flow_min_slider, flow_default_max, flow_step = (
                    flow_ceiling_slider_settings(observed_flow_max)
                )
                flow_max = st.slider(
                    "Maximum conduit flow in flow plots [m³/s]",
                    min_value=float(flow_min_slider),
                    max_value=float(flow_default_max),
                    value=float(flow_default_max),
                    step=float(flow_step),
                    format="%.2e",
                    key=f"conduit_flow_ceiling_{run_key}",
                    help=(
                        "The default is based on the maximum actual tube or external "
                        "conduit-boundary flow over the complete run. Lower it to focus on smaller "
                        "flow variations. The signed exchange-flow plots retain a "
                        "separate fixed full-run scale so their sign and magnitude "
                        "remain directly comparable when time or node is changed."
                    ),
                )
                exchange_limit = global_exchange_flow_limit(diagnostics)

            # -----------------------------------------------------------------
            # Head diagnostics — each section in its own expander
            # -----------------------------------------------------------------
            if show_head_diagnostics:
                assert head_ylim is not None and matrix_head_max is not None

                with st.expander(
                    "Head diagnostics — 1. Matrix head in plan view",
                    expanded=True,
                ):
                    st.caption(
                        "Filled contours show the matrix hydraulic-head field. The "
                        "full conduit is shown in the run color; the selected node is "
                        "surrounded by a complementary ring and its perpendicular "
                        "section is marked by the dashed line. The selected tube is "
                        "shown as two parallel lines joining its two end nodes."
                    )
                    try:
                        fig_plan = make_plan_view_head_plot_v7(
                            diagnostics,
                            int(selected_time_index),
                            conduit_color=conduit_color,
                            selected_node=int(selected_node),
                            selected_tube=int(selected_tube),
                            matrix_head_max=float(matrix_head_max),
                            comparison_node=(
                                int(comparison_node)
                                if comparison_mode and comparison_node is not None
                                else None
                            ),
                            comparison_tube=(
                                int(comparison_tube)
                                if comparison_mode and comparison_tube is not None
                                else None
                            ),
                        )
                        st.pyplot(fig_plan, use_container_width=True)
                        plt.close(fig_plan)
                    except Exception as exc:
                        st.warning(f"Plan-view head plot could not be generated: {exc}")

                with st.expander(
                    "Head diagnostics — 2. Heads along and perpendicular to the conduit",
                    expanded=False,
                ):
                    st.caption(
                        "The base curves use the shared base time/node. When comparison "
                        "mode is active, dashed curves add the second time and the second "
                        "perpendicular section uses the comparison node."
                    )
                    fig_long = make_longitudinal_head_plot_v6(
                        diagnostics,
                        int(selected_time_index),
                        conduit_color=conduit_color,
                        selected_node=int(selected_node),
                        head_ylim=head_ylim,
                        comparison_time_index=(
                            int(comparison_time_index)
                            if comparison_mode and comparison_time_index is not None
                            else None
                        ),
                        comparison_node=(
                            int(comparison_node)
                            if comparison_mode and comparison_node is not None
                            else None
                        ),
                    )
                    st.pyplot(fig_long, use_container_width=True)
                    plt.close(fig_long)

                    fig_cross = make_perpendicular_head_plot_v6(
                        diagnostics,
                        int(selected_time_index),
                        selected_node=int(selected_node),
                        conduit_color=conduit_color,
                        head_ylim=head_ylim,
                        comparison_time_index=(
                            int(comparison_time_index)
                            if comparison_mode and comparison_time_index is not None
                            else None
                        ),
                        comparison_node=(
                            int(comparison_node)
                            if comparison_mode and comparison_node is not None
                            else None
                        ),
                    )
                    st.pyplot(fig_cross, use_container_width=True)
                    plt.close(fig_cross)

                with st.expander(
                    "Head diagnostics — 3. Transient head development at the conduit node",
                    expanded=False,
                ):
                    st.caption(
                        f"Transient conduit and matrix heads are shown for base node "
                        f"{selected_node}. In comparison mode, the second node is added "
                        "with dashed curves and both diagnostic times are marked."
                    )
                    fig_transient = make_node_head_timeseries_plot_v6(
                        diagnostics,
                        int(selected_node),
                        conduit_color=conduit_color,
                        selected_time_index=int(selected_time_index),
                        head_ylim=head_ylim,
                        comparison_node=(
                            int(comparison_node)
                            if comparison_mode and comparison_node is not None
                            else None
                        ),
                        comparison_time_index=(
                            int(comparison_time_index)
                            if comparison_mode and comparison_time_index is not None
                            else None
                        ),
                    )
                    st.pyplot(fig_transient, use_container_width=True)
                    plt.close(fig_transient)

                consistency = diagnostics.get("matrix_head_consistency_max_abs_diff")
                if consistency is not None and np.isfinite(consistency):
                    st.caption(
                        "Post-processing consistency check: maximum difference between "
                        "listing-file matrix heads and binary MODFLOW heads at conduit "
                        f"cells = {consistency:.2e} m."
                    )

            # -----------------------------------------------------------------
            # Flow diagnostics — analogous spatial and transient sections
            # -----------------------------------------------------------------
            if show_flow_diagnostics:
                assert flow_max is not None and exchange_limit is not None

                with st.expander(
                    "Flow diagnostics — 1. Flow and exchange along the conduit",
                    expanded=True,
                ):
                    st.caption(
                        "Conduit flow uses the actual CFP tube Q values. The profile also "
                        "shows the external boundary fluxes explicitly: spring outflow from "
                        "negative QFIX and sinkhole inflow from DIRECT RECHARGE. "
                        "Matrix–conduit exchange remains node based and uses CFP's signed "
                        "EXCHANGE value. In the conduit-flow plot, square markers denote "
                        "tube-flow values. The active tube is connected between its two end "
                        "nodes; the selected node is filled and the opposite tube end is open. "
                        "Exchange values are marked with circles because they belong to nodes."
                    )
                    fig_flow_profile = make_conduit_flow_profile_plot_v7(
                        diagnostics,
                        int(selected_time_index),
                        conduit_color=conduit_color,
                        selected_node=int(selected_node),
                        selected_tube=int(selected_tube),
                        flow_max=float(flow_max),
                        comparison_time_index=(
                            int(comparison_time_index)
                            if comparison_mode and comparison_time_index is not None
                            else None
                        ),
                        comparison_node=(
                            int(comparison_node)
                            if comparison_mode and comparison_node is not None
                            else None
                        ),
                        comparison_tube=(
                            int(comparison_tube)
                            if comparison_mode and comparison_tube is not None
                            else None
                        ),
                    )
                    st.pyplot(fig_flow_profile, use_container_width=True)
                    plt.close(fig_flow_profile)

                    fig_exchange_profile = make_exchange_flow_profile_plot_v7(
                        diagnostics,
                        int(selected_time_index),
                        conduit_color=conduit_color,
                        selected_node=int(selected_node),
                        selected_tube=int(selected_tube),
                        exchange_limit=float(exchange_limit),
                        comparison_time_index=(
                            int(comparison_time_index)
                            if comparison_mode and comparison_time_index is not None
                            else None
                        ),
                        comparison_node=(
                            int(comparison_node)
                            if comparison_mode and comparison_node is not None
                            else None
                        ),
                        comparison_tube=(
                            int(comparison_tube)
                            if comparison_mode and comparison_tube is not None
                            else None
                        ),
                    )
                    st.pyplot(fig_exchange_profile, use_container_width=True)
                    plt.close(fig_exchange_profile)

                with st.expander(
                    "Flow diagnostics — 2. Transient tube flow and node exchange",
                    expanded=False,
                ):
                    st.caption(
                        f"Actual tube flow is shown through time for base tube "
                        f"{selected_tube}; signed matrix–conduit exchange is shown for "
                        f"base node {selected_node}. Comparison mode adds the second tube "
                        "and second node and marks both diagnostic times."
                    )
                    fig_flow_transient = make_tube_flow_timeseries_plot_v7(
                        diagnostics,
                        int(selected_tube),
                        conduit_color=conduit_color,
                        selected_time_index=int(selected_time_index),
                        flow_max=float(flow_max),
                        comparison_tube=(
                            int(comparison_tube)
                            if comparison_mode and comparison_tube is not None
                            else None
                        ),
                        comparison_time_index=(
                            int(comparison_time_index)
                            if comparison_mode and comparison_time_index is not None
                            else None
                        ),
                    )
                    st.pyplot(fig_flow_transient, use_container_width=True)
                    plt.close(fig_flow_transient)

                    fig_exchange_transient = make_node_exchange_flow_timeseries_plot_v6(
                        diagnostics,
                        int(selected_node),
                        conduit_color=conduit_color,
                        selected_time_index=int(selected_time_index),
                        exchange_limit=float(exchange_limit),
                        comparison_node=(
                            int(comparison_node)
                            if comparison_mode and comparison_node is not None
                            else None
                        ),
                        comparison_time_index=(
                            int(comparison_time_index)
                            if comparison_mode and comparison_time_index is not None
                            else None
                        ),
                    )
                    st.pyplot(fig_exchange_transient, use_container_width=True)
                    plt.close(fig_exchange_transient)

    with st.expander("Show parameters of current run"):
        st.dataframe(
            parameter_table(current["params"]),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# Comparison section
# =============================================================================
# -----------------------------------------------------------------------------
# 5.4 Stored-run comparison
# -----------------------------------------------------------------------------
st.divider()
st.header("2. Compare stored runs")

saved = st.session_state.saved_scenarios

if not saved:
    st.caption(
        "No completed runs yet. Each successful model execution will be stored "
        f"automatically; up to {MAX_STORED_RUNS} runs are retained."
    )
else:
    all_run_ids = [int(item["execution_number"]) for item in saved]
    runs_by_id = {int(item["execution_number"]): item for item in saved}

    top1, top2 = st.columns([2, 1])
    with top1:
        if "comparison_run_selection" not in st.session_state:
            st.session_state.comparison_run_selection = all_run_ids.copy()
        else:
            # Names are editable, so the multiselect tracks stable execution IDs.
            # Also migrate/reset incompatible selections left by an older source
            # version during Streamlit hot reload.
            try:
                st.session_state.comparison_run_selection = [
                    int(run_id)
                    for run_id in st.session_state.comparison_run_selection
                    if int(run_id) in all_run_ids
                ]
            except (TypeError, ValueError):
                st.session_state.comparison_run_selection = all_run_ids.copy()

        selected_run_ids = st.multiselect(
            "Runs shown in the comparison",
            options=all_run_ids,
            format_func=lambda run_id: runs_by_id[int(run_id)]["name"],
            key="comparison_run_selection",
        )

    with top2:
        time_unit = st.radio(
            "Time axis",
            options=["hours", "seconds"],
            horizontal=True,
        )

    selected_ids = {int(run_id) for run_id in selected_run_ids}
    selected_scenarios = [
        item
        for item in saved
        if int(item["execution_number"]) in selected_ids
    ]

    if selected_scenarios:
        fig_compare = make_comparison_plot(
            selected_scenarios,
            time_unit=time_unit,
        )
        st.pyplot(fig_compare, use_container_width=True)
        plt.close(fig_compare)
    else:
        st.info("Select at least one stored run to display.")

    if selected_scenarios:
        show_budget_comparison = st.toggle(
            "Compare cumulative budgets",
            value=False,
            key="show_budget_comparison",
        )
        if show_budget_comparison:
            budget_scenarios = [
                scenario for scenario in selected_scenarios
                if scenario.get("budget") is not None
            ]
            if budget_scenarios:
                fig_budget_compare = make_budget_comparison_plot(budget_scenarios)
                st.pyplot(fig_budget_compare, use_container_width=True)
                plt.close(fig_budget_compare)
                if len(budget_scenarios) < len(selected_scenarios):
                    st.caption(
                        "At least one selected run has no parsed cumulative budget "
                        "and was omitted from this comparison."
                    )
            else:
                st.caption("No selected run contains cumulative budget data.")

    # -------------------------------------------------------------------------
    # Optional comparison of diagnostic plots from stored runs
    # -------------------------------------------------------------------------
    if selected_scenarios:
        with st.expander("Compare selected head and flow diagnostics", expanded=False):
            st.caption(
                "Stored-run diagnostic comparisons intentionally use one base time and "
                "one base conduit node. The tube is selected as the left or right tube "
                "adjacent to that node. Secondary current-run comparison selections are "
                "not stored or added here."
            )
            diagnostic_scenarios = [
                scenario
                for scenario in selected_scenarios
                if scenario.get("diagnostics") is not None
            ]

            if not diagnostic_scenarios:
                st.caption(
                    "None of the selected runs contains diagnostic data."
                )
            else:
                reference_diagnostics = diagnostic_scenarios[0]["diagnostics"]
                reference_times = np.asarray(reference_diagnostics["times"], dtype=float)
                reference_nodes = np.asarray(
                    reference_diagnostics["node_numbers"], dtype=int
                ).tolist()
                default_compare_time_idx = int(
                    np.argmin(np.abs(reference_times - (1.0 + 2.0 * 3600.0)))
                )

                if "comparison_diagnostic_time" not in st.session_state:
                    st.session_state.comparison_diagnostic_time = default_compare_time_idx
                if (
                    "comparison_diagnostic_node" not in st.session_state
                    or st.session_state.comparison_diagnostic_node not in reference_nodes
                ):
                    st.session_state.comparison_diagnostic_node = reference_nodes[-1]

                compare_control_a, compare_control_b, compare_control_c = st.columns(3)
                with compare_control_a:
                    compare_time_idx = st.select_slider(
                        "Comparison diagnostic time",
                        options=list(range(len(reference_times))),
                        format_func=lambda idx: format_elapsed_time(
                            float(reference_times[idx])
                        ),
                        key="comparison_diagnostic_time",
                    )
                with compare_control_b:
                    compare_node = st.select_slider(
                        "Comparison conduit node",
                        options=reference_nodes,
                        key="comparison_diagnostic_node",
                    )

                stored_tube_choices = adjacent_tubes_for_node(
                    reference_diagnostics, int(compare_node)
                )
                stored_sides = list(stored_tube_choices)
                stored_side_key = "comparison_diagnostic_tube_side"
                if (
                    stored_side_key not in st.session_state
                    or st.session_state[stored_side_key] not in stored_sides
                ):
                    st.session_state[stored_side_key] = default_tube_side(
                        reference_diagnostics, int(compare_node)
                    )
                with compare_control_c:
                    compare_tube_side = st.radio(
                        "Adjacent comparison tube",
                        options=stored_sides,
                        horizontal=True,
                        key=stored_side_key,
                        format_func=lambda side: (
                            f"{side} (tube {stored_tube_choices[side]})"
                        ),
                    )
                compare_tube = int(stored_tube_choices[compare_tube_side])

                target_compare_time = float(reference_times[int(compare_time_idx)])

                comparison_plot_options = [
                    "Heads along conduit",
                    "Transient heads at selected node",
                    "Flow and exchange along conduit",
                    "Transient tube flow and node exchange",
                ]
                chosen_diagnostic_plots = st.multiselect(
                    "Diagnostic plots to compare",
                    options=comparison_plot_options,
                    default=[],
                    key="comparison_diagnostic_plots",
                    help=(
                        "Select only the diagnostics needed for the current comparison. "
                        "Each stored run keeps its own run color; head comparisons use "
                        "the selected node, while flow comparisons use the selected tube "
                        "for tube Q and the selected node for matrix–conduit exchange."
                    ),
                )

                head_scenarios = [
                    scenario
                    for scenario in diagnostic_scenarios
                    if all(
                        key in scenario["diagnostics"]
                        for key in ("conduit_heads", "matrix_heads_at_nodes")
                    )
                ]
                flow_scenarios = [
                    scenario
                    for scenario in diagnostic_scenarios
                    if all(
                        key in scenario["diagnostics"]
                        for key in (
                            "tube_flow",
                            "direct_recharge_total",
                            "spring_outflow",
                            "exchange_flow",
                        )
                    )
                ]

                if "Heads along conduit" in chosen_diagnostic_plots:
                    st.markdown("##### Heads along conduit")
                    if head_scenarios:
                        fig = make_head_profile_comparison_plot(
                            head_scenarios,
                            target_time=target_compare_time,
                            node_number=int(compare_node),
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.caption("Head-profile data are unavailable for the selected runs.")

                if "Transient heads at selected node" in chosen_diagnostic_plots:
                    st.markdown("##### Transient heads at selected node")
                    if head_scenarios:
                        fig = make_head_timeseries_comparison_plot(
                            head_scenarios,
                            node_number=int(compare_node),
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.caption("Transient head data are unavailable for the selected runs.")

                if "Flow and exchange along conduit" in chosen_diagnostic_plots:
                    st.markdown("##### Flow and exchange along conduit")
                    if flow_scenarios:
                        fig_q, fig_ex = make_flow_profile_comparison_plots_v7(
                            flow_scenarios,
                            target_time=target_compare_time,
                            node_number=int(compare_node),
                            tube_number=int(compare_tube),
                        )
                        st.pyplot(fig_q, use_container_width=True)
                        plt.close(fig_q)
                        st.pyplot(fig_ex, use_container_width=True)
                        plt.close(fig_ex)
                    else:
                        st.caption(
                            "Flow diagnostics are unavailable for these stored runs. "
                            "Runs created with an older app version may need to be rerun."
                        )

                if "Transient tube flow and node exchange" in chosen_diagnostic_plots:
                    st.markdown("##### Transient tube flow and node exchange")
                    if flow_scenarios:
                        fig_q, fig_ex = make_flow_timeseries_comparison_plots_v7(
                            flow_scenarios,
                            node_number=int(compare_node),
                            tube_number=int(compare_tube),
                        )
                        st.pyplot(fig_q, use_container_width=True)
                        plt.close(fig_q)
                        st.pyplot(fig_ex, use_container_width=True)
                        plt.close(fig_ex)
                    else:
                        st.caption(
                            "Flow diagnostics are unavailable for these stored runs. "
                            "Runs created with an older app version may need to be rerun."
                        )

                if len(diagnostic_scenarios) < len(selected_scenarios):
                    st.caption(
                        "At least one selected run did not contain diagnostic output and "
                        "was omitted from the diagnostic comparison."
                    )

    settings_rows = []
    for scenario in saved:
        row = {
            "Run": scenario["name"],
            "Execution": scenario["execution_number"],
            "Storage slot": scenario["slot_number"],
            **scenario["params"],
        }
        settings_rows.append(row)
    settings_df = pd.DataFrame(settings_rows)

    st.markdown("#### Stored parameter sets")
    st.caption(
        "The table is ordered from the oldest retained execution to the newest. "
        "After 5 executions, storage slots are reused cyclically. User-defined "
        "run names remain unchanged until their corresponding slot is overwritten."
    )
    st.dataframe(settings_df, use_container_width=True, hide_index=True)

    csv = settings_df.to_csv(index=False).encode("utf-8")
    b1, b2 = st.columns([1, 1])
    with b1:
        st.download_button(
            "⬇️ Download settings as CSV",
            data=csv,
            file_name="karst_spring_runs.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with b2:
        if st.button(
            "🗑️ Clear all results",
            use_container_width=True,
        ):
            st.session_state.saved_scenarios = []
            st.session_state.current_run = None
            st.session_state.total_run_count = 0
            for key in (
                "comparison_run_selection",
                "comparison_diagnostic_time",
                "comparison_diagnostic_node",
                "comparison_diagnostic_tube_side",
                "comparison_diagnostic_plots",
                "show_budget_comparison",
            ):
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
