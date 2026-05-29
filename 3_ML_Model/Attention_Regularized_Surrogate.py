import os
import re
import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ============================================================
# Paths relative to this script
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MLDATA_DIR = os.path.join(SCRIPT_DIR, "MLDATA")
DEFAULT_DIRS = {
    "TEMP": os.path.join(MLDATA_DIR, "TEMP"),
    "COMP": os.path.join(MLDATA_DIR, "COMP"),
    "ETALIQ": os.path.join(MLDATA_DIR, "ETALIQ"),
    "VEL": os.path.join(MLDATA_DIR, "VEL"),
}

ELEMENTS = ["Co", "Cr", "Fe", "Ni"]

# cTF table used to map existing filenames and target dropdown choices
# to the initial phase-specific compositions. The preferred source is
# nomenclature_phase_composition_thermodynamic_factor_tensor_components.csv
# located beside this script. ETA = 1 is treated as LIQUID and ETA = 0 as FCC.
FALLBACK_CTF_TABLE: Dict[int, Dict[str, List[float]]] = {
    0: {"liquid": [0.35, 0.13, 0.15, 0.37], "fcc": [0.28, 0.18, 0.22, 0.32]},
    1: {"liquid": [0.32, 0.15, 0.17, 0.36], "fcc": [0.25, 0.21, 0.25, 0.29]},
    2: {"liquid": [0.30, 0.19, 0.11, 0.40], "fcc": [0.25, 0.23, 0.20, 0.32]},
    3: {"liquid": [0.33, 0.11, 0.23, 0.33], "fcc": [0.31, 0.15, 0.27, 0.27]},
    4: {"liquid": [0.38, 0.17, 0.19, 0.26], "fcc": [0.30, 0.22, 0.26, 0.22]},
}


def load_ctf_table_from_csv() -> Dict[int, Dict[str, List[float]]]:
    csv_path = os.path.join(SCRIPT_DIR, "nomenclature_phase_composition_thermodynamic_factor_tensor_components.csv")
    if not os.path.exists(csv_path):
        return FALLBACK_CTF_TABLE

    try:
        df = pd.read_csv(csv_path)
        required_cols = {"cTF", "ETA", "Co", "Cr", "Fe", "Ni"}
        if not required_cols.issubset(df.columns):
            return FALLBACK_CTF_TABLE

        table: Dict[int, Dict[str, List[float]]] = {}
        for ctf_idx, group in df.groupby("cTF"):
            ctf_idx = int(ctf_idx)
            liq_rows = group[group["ETA"].astype(float) == 1.0]
            fcc_rows = group[group["ETA"].astype(float) == 0.0]
            if liq_rows.empty or fcc_rows.empty:
                continue
            liq = liq_rows.iloc[0][ELEMENTS].astype(float).to_numpy().tolist()
            fcc = fcc_rows.iloc[0][ELEMENTS].astype(float).to_numpy().tolist()
            table[ctf_idx] = {"liquid": liq, "fcc": fcc}

        return table if table else FALLBACK_CTF_TABLE
    except Exception:
        return FALLBACK_CTF_TABLE


CTF_TABLE: Dict[int, Dict[str, List[float]]] = load_ctf_table_from_csv()

CMAP_OPTIONS = [
    "inferno", "magma", "plasma", "viridis", "cividis", "turbo", "rainbow", "jet",
    "hot", "afmhot", "coolwarm", "RdBu_r", "RdYlBu_r", "Spectral_r",
    "PuBuGn", "YlOrRd", "YlGnBu", "GnBu", "Blues", "Greens",
    "gray", "binary", "bone", "terrain", "nipy_spectral",
]

# ============================================================
# Streamlit page
# ============================================================
st.set_page_config(page_title="Phase-Aware Multi-Field Attention Interpolator", layout="wide")
st.title("Phase-Aware Attention Interpolation for TEMP, COMP, ETALIQ and VEL")
st.markdown(
    """
This app predicts laser-processing fields from `.npy` files stored in `MLDATA/TEMP`, `MLDATA/ETALIQ`, `MLDATA/VEL`, and `MLDATA/COMP`, with filenames like `p350s45cTF0.npy`, where
`p350` is laser power, `s45` is scan speed in cm/s, and `cTF0` identifies the source
initial composition set. The target composition can be selected from the cTF0–cTF4
CSV table, or entered manually as `cTF_New`; Ni is inferred from mass conservation.
"""
)

# ============================================================
# Utilities
# ============================================================
def parse_tensor_filename(filename: str) -> Optional[Dict]:
    match = re.search(r"p(\d+)s(\d+)cTF(\d+)\.npy$", filename, re.IGNORECASE)
    if not match:
        return None
    power = int(match.group(1))
    speed_cm_s = int(match.group(2))
    tf_idx = int(match.group(3))
    if tf_idx not in CTF_TABLE:
        return None
    return {
        "P": power,
        "v_cm_s": speed_cm_s,
        "v": speed_cm_s / 100.0,
        "TF_idx": tf_idx,
        "composition_liquid": np.array(CTF_TABLE[tf_idx]["liquid"], dtype=float),
        "composition_fcc": np.array(CTF_TABLE[tf_idx]["fcc"], dtype=float),
    }


def standardize_field_array(arr: np.ndarray, field_key: str) -> np.ndarray:
    """Return a plottable scalar field. Velocity vector fields are converted to magnitude."""
    arr = np.asarray(arr)

    if field_key == "VEL":
        # Supported vector shapes: (Nt, Ny, Nx, 2/3) or (Nt, 2/3, Ny, Nx)
        if arr.ndim == 4 and arr.shape[-1] in (2, 3):
            arr = np.linalg.norm(arr, axis=-1)
        elif arr.ndim == 4 and arr.shape[1] in (2, 3):
            arr = np.linalg.norm(arr, axis=1)

    return arr


def get_component_view(arr: np.ndarray, field_key: str, component_index: int) -> np.ndarray:
    """Extract one scalar component from 3D/4D fields for plotting/interpolation display."""
    arr = np.asarray(arr)
    arr = standardize_field_array(arr, field_key)

    if arr.ndim == 3:
        return arr

    # Composition files may be stored as (Nt, Ny, Nx, 4) or (Nt, 4, Ny, Nx)
    if arr.ndim == 4:
        if arr.shape[-1] in (3, 4, 5):
            component_index = min(component_index, arr.shape[-1] - 1)
            return arr[..., component_index]
        if arr.shape[1] in (3, 4, 5):
            component_index = min(component_index, arr.shape[1] - 1)
            return arr[:, component_index, :, :]

    raise ValueError(f"Expected 3D scalar field or supported 4D component/vector field, got shape {arr.shape}.")


def load_sources_from_folder(folder_path: str, field_key: str) -> Tuple[List[Dict], List[np.ndarray], List[str], int]:
    sources, arrays, warnings = [], [], []
    if not os.path.isdir(folder_path):
        return sources, arrays, [f"Folder not found: {folder_path}"], 0

    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".npy")])
    for filename in files:
        parsed = parse_tensor_filename(filename)
        if parsed is None:
            warnings.append(f"Skipped {filename}: filename does not match p<power>s<speed>cTF<idx>.npy")
            continue
        try:
            arr = np.load(os.path.join(folder_path, filename))
            arr = standardize_field_array(arr, field_key)
            if arr.ndim not in (3, 4):
                warnings.append(f"Skipped {filename}: expected 3D/4D array, got {arr.shape}")
                continue
            sources.append({
                "file_name": filename,
                "P": parsed["P"],
                "v": parsed["v"],
                "v_cm_s": parsed["v_cm_s"],
                "TF_idx": parsed["TF_idx"],
                "composition_liquid": parsed["composition_liquid"],
                "composition_fcc": parsed["composition_fcc"],
                "shape": arr.shape,
            })
            arrays.append(arr)
        except Exception as exc:
            warnings.append(f"Skipped {filename}: {exc}")
    return sources, arrays, warnings, len(files)


def load_sources_from_upload(uploaded_files, field_key: str) -> Tuple[List[Dict], List[np.ndarray], List[str], int]:
    sources, arrays, warnings = [], [], []
    files = uploaded_files or []
    for uploaded_file in files:
        parsed = parse_tensor_filename(uploaded_file.name)
        if parsed is None:
            warnings.append(f"Skipped {uploaded_file.name}: filename does not match p<power>s<speed>cTF<idx>.npy")
            continue
        try:
            uploaded_file.seek(0)
            arr = np.load(uploaded_file)
            arr = standardize_field_array(arr, field_key)
            if arr.ndim not in (3, 4):
                warnings.append(f"Skipped {uploaded_file.name}: expected 3D/4D array, got {arr.shape}")
                continue
            sources.append({
                "file_name": uploaded_file.name,
                "P": parsed["P"],
                "v": parsed["v"],
                "v_cm_s": parsed["v_cm_s"],
                "TF_idx": parsed["TF_idx"],
                "composition_liquid": parsed["composition_liquid"],
                "composition_fcc": parsed["composition_fcc"],
                "shape": arr.shape,
            })
            arrays.append(arr)
        except Exception as exc:
            warnings.append(f"Skipped {uploaded_file.name}: {exc}")
    return sources, arrays, warnings, len(files)


def sources_dataframe(sources: List[Dict]) -> pd.DataFrame:
    rows = []
    for s in sources:
        cL = np.asarray(s["composition_liquid"], dtype=float)
        cF = np.asarray(s["composition_fcc"], dtype=float)
        rows.append({
            "File": s["file_name"],
            "P (W)": s["P"],
            "v (cm/s)": s["v_cm_s"],
            "cTF": s["TF_idx"],
            "LIQ Co": cL[0],
            "LIQ Cr": cL[1],
            "LIQ Fe": cL[2],
            "LIQ Ni": cL[3],
            "FCC Co": cF[0],
            "FCC Cr": cF[1],
            "FCC Fe": cF[2],
            "FCC Ni": cF[3],
            "Shape": " × ".join(map(str, s["shape"])),
        })
    return pd.DataFrame(rows)


def npy_bytes(arr: np.ndarray) -> io.BytesIO:
    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    return buffer


def npz_bytes(**kwargs) -> io.BytesIO:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **kwargs)
    buffer.seek(0)
    return buffer


def mpl_to_plotly_colorscale(cmap_name: str, n: int = 255):
    cmap = mpl.colormaps.get_cmap(cmap_name)
    return [[i / (n - 1), mpl.colors.rgb2hex(cmap(i / (n - 1)))] for i in range(n)]


def choose_folder_dialog(initial_dir: str) -> Optional[str]:
    """Open a native folder picker when Streamlit is running on a local desktop.

    On headless/HPC sessions without a display, this returns None and the text box
    remains the fallback.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        selected = filedialog.askdirectory(initialdir=initial_dir or SCRIPT_DIR)
        root.destroy()
        return selected or None
    except Exception as exc:
        st.warning(f"Folder picker could not be opened in this session: {exc}")
        return None

# ============================================================
# Attention model based on colleague's phase-aware formulation
# ============================================================
class PhaseAwareAttentionInterpolator(nn.Module):
    def __init__(self, sigma_param=0.20, sigma_comp=0.05, num_heads=4, d_head=8, comp_strength=1.0):
        super().__init__()
        self.sigma_param = sigma_param
        self.sigma_comp = sigma_comp
        self.num_heads = num_heads
        self.d_head = d_head
        self.comp_strength = comp_strength
        self.W_q_proc = nn.Linear(2, num_heads * d_head, bias=False)
        self.W_k_proc = nn.Linear(2, num_heads * d_head, bias=False)

    @staticmethod
    def normalize_params(params_list, target_params):
        all_params = np.array(params_list + [target_params], dtype=float)
        mins = all_params.min(axis=0)
        maxs = all_params.max(axis=0)
        ranges = maxs - mins + 1e-8
        src = (np.array(params_list, dtype=float) - mins) / ranges
        tgt = (np.array(target_params, dtype=float) - mins) / ranges
        return torch.tensor(src, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

    def composition_similarity(self, source_comp: np.ndarray, target_comp: np.ndarray) -> float:
        source_vec = np.asarray(source_comp, dtype=float).ravel()
        target_vec = np.asarray(target_comp, dtype=float).ravel()
        diff2 = float(np.sum((source_vec - target_vec) ** 2))
        sim = np.exp(-diff2 / (2 * self.sigma_comp ** 2))
        return float(sim)

    @staticmethod
    def phase_composition_vector(item: Dict) -> np.ndarray:
        """Concatenate LIQUID and FCC compositions for phase-aware comparison."""
        liquid = np.asarray(item.get("composition_liquid", item.get("composition")), dtype=float)
        fcc = np.asarray(item.get("composition_fcc", liquid), dtype=float)
        return np.concatenate([liquid, fcc])

    def compute_weights(self, sources: List[Dict], target: Dict, use_composition: bool = True) -> Dict[str, np.ndarray]:
        n = len(sources)
        source_proc = [(s["P"], s["v"]) for s in sources]
        target_proc = (target["P"], target["v"])
        src_proc_tensor, tgt_proc_1d = self.normalize_params(source_proc, target_proc)
        tgt_proc = tgt_proc_1d.unsqueeze(0)

        q = self.W_q_proc(tgt_proc).view(1, self.num_heads, self.d_head)
        k = self.W_k_proc(src_proc_tensor).view(n, self.num_heads, self.d_head)
        logits = torch.einsum("nhd,mhd->nmh", k, q) / np.sqrt(self.d_head)
        attn = torch.softmax(logits.squeeze(1), dim=0).mean(dim=1)

        dists = torch.sqrt(torch.sum((src_proc_tensor - tgt_proc) ** 2, dim=1))
        gaussian = torch.exp(-(dists ** 2) / (2 * self.sigma_param ** 2))
        gaussian = gaussian / (gaussian.sum() + 1e-8)

        if use_composition:
            target_comp_vec = self.phase_composition_vector(target)
            sims = np.array([
                self.composition_similarity(self.phase_composition_vector(s), target_comp_vec)
                for s in sources
            ], dtype=float)
            # comp_strength=0 => uniform composition effect; comp_strength=1 => full Gaussian similarity.
            sims = np.power(sims + 1e-12, self.comp_strength)
            comp = torch.tensor(sims / (sims.sum() + 1e-12), dtype=torch.float32)
        else:
            comp = torch.ones(n, dtype=torch.float32) / n

        combined = attn * gaussian * comp
        combined = combined / (combined.sum() + 1e-8)

        return {
            "attention_weights_proc": attn.detach().numpy(),
            "spatial_weights_proc": gaussian.detach().numpy(),
            "composition_weights": comp.detach().numpy(),
            "combined_weights": combined.detach().numpy(),
            "norm_sources_proc": src_proc_tensor.numpy(),
            "norm_target_proc": tgt_proc_1d.numpy(),
            "W_q_proc": self.W_q_proc.weight.data.numpy(),
            "W_k_proc": self.W_k_proc.weight.data.numpy(),
        }

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Target parameters")
    p_target = st.number_input("Power, P (W)", min_value=300.0, max_value=800.0, value=370.0, step=10.0)
    v_target_cm_s = st.number_input("Scan speed, v (cm/s)", min_value=40.0, max_value=80.0, value=50.0, step=5.0)

    ctf_options = [f"cTF{k}" for k in sorted(CTF_TABLE.keys())] + ["cTF_New"]
    target_ctf_choice = st.selectbox("Target composition set", ctf_options, index=0)

    if target_ctf_choice != "cTF_New":
        selected_ctf = int(target_ctf_choice.replace("cTF", ""))
        liq_vals = np.array(CTF_TABLE[selected_ctf]["liquid"], dtype=float)
        fcc_vals = np.array(CTF_TABLE[selected_ctf]["fcc"], dtype=float)

        co_target, cr_target, fe_target, ni_target = liq_vals.tolist()
        co_fcc_target, cr_fcc_target, fe_fcc_target, ni_fcc_target = fcc_vals.tolist()
        target_valid = True

        st.markdown("**LIQUID composition selected from CSV**")
        st.dataframe(
            pd.DataFrame([dict(zip(ELEMENTS, liq_vals))], index=[target_ctf_choice]),
            width='stretch',
        )
        st.markdown("**FCC composition selected from CSV**")
        st.dataframe(
            pd.DataFrame([dict(zip(ELEMENTS, fcc_vals))], index=[target_ctf_choice]),
            width='stretch',
        )
    else:
        st.markdown("**Custom LIQUID composition**")
        co_target = st.slider("LIQ Co", 0.00, 1.00, 0.35, 0.01)
        cr_target = st.slider("LIQ Cr", 0.00, 1.00, 0.13, 0.01)
        fe_target = st.slider("LIQ Fe", 0.00, 1.00, 0.15, 0.01)
        ni_target = round(1.0 - co_target - cr_target - fe_target, 4)

        st.markdown("**Custom FCC composition**")
        co_fcc_target = st.slider("FCC Co", 0.00, 1.00, 0.28, 0.01)
        cr_fcc_target = st.slider("FCC Cr", 0.00, 1.00, 0.18, 0.01)
        fe_fcc_target = st.slider("FCC Fe", 0.00, 1.00, 0.22, 0.01)
        ni_fcc_target = round(1.0 - co_fcc_target - cr_fcc_target - fe_fcc_target, 4)

        liq_valid = 0.0 <= ni_target <= 1.0
        fcc_valid = 0.0 <= ni_fcc_target <= 1.0
        target_valid = liq_valid and fcc_valid

        if liq_valid:
            st.success(f"LIQ Ni = 1 - Co - Cr - Fe = {ni_target:.3f}")
        else:
            st.error(f"Invalid LIQ composition: Ni = {ni_target:.3f}. Adjust LIQ Co, Cr or Fe.")

        if fcc_valid:
            st.success(f"FCC Ni = 1 - Co - Cr - Fe = {ni_fcc_target:.3f}")
        else:
            st.error(f"Invalid FCC composition: Ni = {ni_fcc_target:.3f}. Adjust FCC Co, Cr or Fe.")

    target_composition = np.array([co_target, cr_target, fe_target, ni_target], dtype=float)
    target_composition_fcc = np.array([co_fcc_target, cr_fcc_target, fe_fcc_target, ni_fcc_target], dtype=float)

    st.header("Attention model")
    sigma_param = st.slider("Gaussian locality σ for P-v", 0.05, 0.50, 0.18, 0.01)
    sigma_comp = st.slider("Composition similarity σ", 0.01, 0.25, 0.05, 0.01)
    comp_strength = st.slider("Composition influence strength", 0.0, 2.0, 1.0, 0.1)
    num_heads = st.slider("Attention heads", 1, 12, 8, 1)
    d_head = st.slider("Dimension per head", 4, 16, 6, 1)
    seed = st.number_input("Random seed", 0, 9999, 42, 1)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    st.header("Colormaps")
    temp_cmap = st.selectbox("TEMP colormap", CMAP_OPTIONS, index=6)
    comp_cmap = st.selectbox("COMP colormap", CMAP_OPTIONS, index=3)
    phase_cmap = st.selectbox("ETALIQ colormap", CMAP_OPTIONS, index=11)
    vel_cmap = st.selectbox("VEL colormap", CMAP_OPTIONS, index=5)

    st.header("Phase display")
    phase_threshold = st.slider("LIQUID threshold", 0.01, 0.99, 0.50, 0.01)

    with st.expander("Resolved local folders", expanded=False):
        st.code(f"MLDATA_DIR: {MLDATA_DIR}")
        for key, path in DEFAULT_DIRS.items():
            st.code(f"{key}: {path}")

TARGET = {
    "P": float(p_target),
    "v_cm_s": float(v_target_cm_s),
    "v": float(v_target_cm_s) / 100.0,
    # Backward-compatible alias used by older code sections: LIQUID composition.
    "composition": target_composition,
    "composition_liquid": target_composition,
    "composition_fcc": target_composition_fcc,
    "ctf_choice": target_ctf_choice,
}

# ============================================================
# Common tab function
# ============================================================
def run_field_tab(field_key: str, label: str, unit: str, default_folder: str, cmap_name: str, is_phase: bool = False):
    st.subheader(f"{label} prediction")

    folder_col, browse_col, method_col = st.columns([1.65, 0.35, 1])
    folder_key = f"folder_{field_key}"
    if folder_key not in st.session_state:
        st.session_state[folder_key] = default_folder
    with folder_col:
        folder_path = st.text_input(f"{label} folder", key=folder_key)
    with browse_col:
        st.write("")
        st.write("")
        if st.button("Browse", key=f"browse_{field_key}", help="Open a local folder picker when available"):
            selected_folder = choose_folder_dialog(st.session_state[folder_key])
            if selected_folder:
                st.session_state[folder_key] = selected_folder
                st.rerun()
    with method_col:
        load_method = st.selectbox("Loading method", ["From local folder", "Upload files"], key=f"load_{field_key}")

    if load_method == "From local folder":
        if not os.path.isdir(folder_path):
            st.error(f"Folder not found: `{folder_path}`. Expected data inside `MLDATA/{field_key}/`. No folder was created.")
            sources, arrays_raw, warnings, file_count = [], [], [], 0
        else:
            sources, arrays_raw, warnings, file_count = load_sources_from_folder(folder_path, field_key)
            st.info(f"Found {file_count} .npy files in {label} (MLDATA/{field_key}/): `{folder_path}`")
    else:
        uploaded = st.file_uploader(
            f"Upload {label} .npy files named p350s45cTF0.npy",
            type=["npy"],
            accept_multiple_files=True,
            key=f"upload_{field_key}",
        )
        sources, arrays_raw, warnings, file_count = load_sources_from_upload(uploaded, field_key)
        st.info(f"Found {file_count} uploaded .npy files for {label}.")

    for w in warnings[:8]:
        st.warning(w)
    if len(warnings) > 8:
        st.warning(f"Additional warnings hidden: {len(warnings) - 8}")

    if sources:
        st.markdown(f"### 📋 Loaded {len(sources)} Valid Source Simulations")
        st.dataframe(sources_dataframe(sources), width='stretch')

    if len(sources) < 2:
        st.info("Load at least two valid source simulations to run interpolation.")
        return

    # Component selector for 4D composition-like arrays.
    component_index = 0
    sample = standardize_field_array(arrays_raw[0], field_key)
    if sample.ndim == 4:
        if sample.shape[-1] in (3, 4, 5):
            max_comp = sample.shape[-1] - 1
            choices = list(range(sample.shape[-1]))
        elif sample.shape[1] in (3, 4, 5):
            max_comp = sample.shape[1] - 1
            choices = list(range(sample.shape[1]))
        else:
            choices = [0]
        element_labels = [ELEMENTS[i] if i < 4 else f"Component {i}" for i in choices]
        selected_label = st.selectbox("Component to display/interpolate", element_labels, key=f"component_{field_key}")
        component_index = element_labels.index(selected_label)

    # Convert all arrays to scalar fields with same view.
    arrays = []
    source_shapes = []
    conversion_errors = []
    for src, arr in zip(sources, arrays_raw):
        try:
            scalar_arr = get_component_view(arr, field_key, component_index)
            arrays.append(scalar_arr)
            source_shapes.append(scalar_arr.shape)
        except Exception as exc:
            conversion_errors.append(f"{src['file_name']}: {exc}")
    for err in conversion_errors:
        st.error(err)
    if conversion_errors:
        return

    if len(set(source_shapes)) != 1:
        st.error(f"All scalar fields must have identical shape. Found: {sorted(set(source_shapes))}")
        return

    use_comp = True
    if field_key == "TEMP":
        use_comp = st.checkbox("Use composition similarity in TEMP weighting", value=True, key="use_comp_TEMP")
    elif field_key == "VEL":
        use_comp = st.checkbox("Use composition similarity in VEL weighting", value=True, key="use_comp_VEL")
    elif field_key == "ETALIQ":
        use_comp = st.checkbox("Use composition similarity in ETALIQ weighting", value=True, key="use_comp_ETALIQ")

    button_disabled = not target_valid
    if st.button(f"Run {label} prediction", type="primary", disabled=button_disabled, key=f"run_{field_key}"):
        with st.spinner(f"Computing phase-aware attention interpolation for {label}..."):
            model = PhaseAwareAttentionInterpolator(
                sigma_param=sigma_param,
                sigma_comp=sigma_comp,
                num_heads=num_heads,
                d_head=d_head,
                comp_strength=comp_strength,
            )
            results = model.compute_weights(sources, TARGET, use_composition=use_comp)
            weights = results["combined_weights"]
            predicted = np.zeros_like(arrays[0], dtype=np.float64)
            for weight, arr in zip(weights, arrays):
                predicted += weight * arr

            if field_key == "ETALIQ":
                predicted = (predicted >= 0.5).astype(np.float64)

            st.session_state[f"pred_{field_key}"] = predicted
            st.session_state[f"res_{field_key}"] = results
            st.session_state[f"src_{field_key}"] = sources
            st.session_state[f"shape_{field_key}"] = predicted.shape

    predicted = st.session_state.get(f"pred_{field_key}")
    results = st.session_state.get(f"res_{field_key}")
    cached_sources = st.session_state.get(f"src_{field_key}")

    if predicted is None or results is None or cached_sources is None:
        return

    Nt, Ny, Nx = predicted.shape
    st.success(f"Prediction available: {predicted.shape} = (Nt, Ny, Nx)")

    st.markdown("### Hybrid attention weights")
    dfw = pd.DataFrame({
        "Source File": [s["file_name"] for s in cached_sources],
        "P (W)": [s["P"] for s in cached_sources],
        "v (cm/s)": [s["v_cm_s"] for s in cached_sources],
        "cTF": [s["TF_idx"] for s in cached_sources],
        "Process attention": np.round(results["attention_weights_proc"], 5),
        "P-v Gaussian": np.round(results["spatial_weights_proc"], 5),
        "Composition similarity": np.round(results["composition_weights"], 5),
        "Final weight": np.round(results["combined_weights"], 5),
    })
    st.dataframe(dfw.style.bar(subset=["Final weight"], color="#5fba7d"), width='stretch')

    with st.expander("Parameter-space weight plot and query/key matrices", expanded=False):
        col_param, col_qk = st.columns([1.45, 1.0])

        src_p = np.array([s["P"] for s in cached_sources], dtype=float)
        src_v = np.array([s["v_cm_s"] for s in cached_sources], dtype=float)
        src_tf = np.array([s["TF_idx"] for s in cached_sources], dtype=float)

        p_min, p_max = min(src_p.min(), float(p_target)), max(src_p.max(), float(p_target))
        v_min, v_max = min(src_v.min(), float(v_target_cm_s)), max(src_v.max(), float(v_target_cm_s))
        p_norm = (src_p - p_min) / (p_max - p_min + 1e-8)
        v_norm = (src_v - v_min) / (v_max - v_min + 1e-8)
        tf_norm = src_tf / 4.0

        source_tf_comps = np.array([s["composition_liquid"] for s in cached_sources], dtype=float)
        unique_tf = sorted(set(int(s["TF_idx"]) for s in cached_sources))
        tf_comp_distance = {}
        for tf in unique_tf:
            tf_vectors = np.array([s["composition_liquid"] for s in cached_sources if int(s["TF_idx"]) == tf], dtype=float)
            tf_mean = tf_vectors.mean(axis=0)
            tf_comp_distance[tf] = float(np.linalg.norm(tf_mean - target_composition))
        nearest_tf = min(tf_comp_distance, key=tf_comp_distance.get) if tf_comp_distance else 0
        target_tf_norm = nearest_tf / 4.0

        with col_param:
            fig_param = go.Figure()
            fig_param.add_trace(go.Scatter3d(
                x=v_norm,
                y=p_norm,
                z=tf_norm,
                mode="markers+text",
                marker=dict(
                    size=7,
                    color=results["combined_weights"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Final weight"),
                    opacity=0.92,
                ),
                text=[f"p{s['P']} s{s['v_cm_s']} cTF{s['TF_idx']}" for s in cached_sources],
                textposition="top center",
                hovertemplate=(
                    "Power=%{customdata[0]} W<br>"
                    "Scan speed=%{customdata[1]} cm/s<br>"
                    "cTF=%{customdata[2]}<br>"
                    "Weight=%{marker.color:.5f}<extra></extra>"
                ),
                customdata=np.column_stack([src_p, src_v, src_tf]),
                name="Sources",
            ))
            fig_param.add_trace(go.Scatter3d(
                x=[(float(v_target_cm_s) - v_min) / (v_max - v_min + 1e-8)],
                y=[(float(p_target) - p_min) / (p_max - p_min + 1e-8)],
                z=[target_tf_norm],
                mode="markers+text",
                marker=dict(size=11, symbol="diamond", color="red"),
                text=[f"Target<br>nearest cTF{nearest_tf}"],
                textposition="bottom center",
                name="Target",
            ))
            fig_param.update_layout(
                title="3D normalized parameter space",
                scene=dict(
                    xaxis_title="Normalized scan speed",
                    yaxis_title="Normalized power",
                    zaxis_title="cTF index / 4",
                    zaxis=dict(tickmode="array", tickvals=[0, 0.25, 0.5, 0.75, 1.0], ticktext=["cTF0", "cTF1", "cTF2", "cTF3", "cTF4"]),
                ),
                margin=dict(l=0, r=0, t=45, b=0),
                height=560,
            )
            st.plotly_chart(fig_param, width='stretch', key=f"param_plot_3d_{field_key}")

        with col_qk:
            fig_qk, (ax_q, ax_k) = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
            im_q = ax_q.imshow(results["W_q_proc"], aspect="auto", cmap="coolwarm")
            ax_q.set_title(r"$W_q$ (Query)", fontsize=12, fontweight="bold")
            ax_q.set_xlabel("Head × Dim")
            ax_q.set_ylabel("Input: P, v")
            fig_qk.colorbar(im_q, ax=ax_q, fraction=0.046, pad=0.04)

            im_k = ax_k.imshow(results["W_k_proc"], aspect="auto", cmap="coolwarm")
            ax_k.set_title(r"$W_k$ (Key)", fontsize=12, fontweight="bold")
            ax_k.set_xlabel("Head × Dim")
            ax_k.set_ylabel("Input: P, v")
            fig_qk.colorbar(im_k, ax=ax_k, fraction=0.046, pad=0.04)
            st.pyplot(fig_qk)

    st.markdown(f"### {label} animation")

    if is_phase:
        anim_field = predicted.astype(float)
        plot_unit = "phase"
        stat_field = predicted
        st.write(
            f"ETALIQ prediction has been binarized using fixed threshold 0.50: "
            f"values >= 0.5 are LIQUID = 1 and values < 0.5 are FCC = 0. "
            f"Binary field statistics: min={stat_field.min():.4f}, "
            f"max={stat_field.max():.4f}, mean={stat_field.mean():.4f}."
        )
    else:
        anim_field = predicted
        plot_unit = unit
        st.write(
            f"Full prediction statistics: min={np.nanmin(predicted):.4f} {unit}, "
            f"max={np.nanmax(predicted):.4f} {unit}, mean={np.nanmean(predicted):.4f} {unit}"
        )

    # Animated Plotly heatmap shown by default.
    # Colorbar limits/ticks are updated frame-by-frame for the active time index.
    colorscale = mpl_to_plotly_colorscale(cmap_name)

    def heatmap_for_time(k: int) -> go.Heatmap:
        z = np.flipud(anim_field[k])
        if is_phase:
            local_min, local_max = 0.0, 1.0
            tickvals = [0, 1]
            ticktext = ["FCC = 0", "LIQUID = 1"]
        else:
            local_min = float(np.nanmin(anim_field[k]))
            local_max = float(np.nanmax(anim_field[k]))
            if np.isclose(local_min, local_max):
                local_max = local_min + 1e-12
            tickvals = np.linspace(local_min, local_max, 5).tolist()
            ticktext = [f"{v:.4g}" for v in tickvals]

        return go.Heatmap(
            z=z,
            colorscale=colorscale,
            zmin=local_min,
            zmax=local_max,
            colorbar=dict(title=plot_unit, tickmode="array", tickvals=tickvals, ticktext=ticktext),
        )

    frames = [go.Frame(data=heatmap_for_time(k), name=f"t{k}") for k in range(Nt)]
    fig = go.Figure(data=heatmap_for_time(0), frames=frames)
    fig.update_layout(
        title=f"{label} evolution",
        xaxis_title="X index",
        yaxis_title="Y index (top = surface)",
        updatemenus=[{
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
            "type": "buttons",
            "x": 0.05,
            "y": 0,
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {"method": "animate", "args": [[f"t{k}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], "label": str(k)}
                for k in range(Nt)
            ],
            "x": 0.1,
            "len": 0.85,
            "y": 0,
            "currentvalue": {"prefix": "Time index: "},
        }],
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, width='stretch', key=f"anim_plot_{field_key}")

    # Static plot inside expander only.
    with st.expander("Static plot", expanded=False):
        static_key = f"static_time_{field_key}"
        if static_key not in st.session_state:
            st.session_state[static_key] = Nt // 2
        st.session_state[static_key] = min(max(int(st.session_state[static_key]), 0), Nt - 1)
        t_idx_static = st.slider("Time index", 0, Nt - 1, key=static_key)
        current = predicted[t_idx_static]

        if is_phase:
            display_field = current.astype(float)
            stat_field = predicted[t_idx_static]
            st.write(
                f"Binary ETALIQ field @ t={t_idx_static}: min={stat_field.min():.4f}, "
                f"max={stat_field.max():.4f}, mean={stat_field.mean():.4f}. "
                f"Binarization used fixed threshold 0.50."
            )
        else:
            display_field = current
            st.write(
                f"Statistics @ t={t_idx_static}: min={current.min():.4f} {unit}, "
                f"max={current.max():.4f} {unit}, mean={current.mean():.4f} {unit}"
            )

        fig_static, ax = plt.subplots(figsize=(9.5, 7.5))
        slice_disp = np.flipud(display_field)

        if is_phase:
            local_min, local_max = 0.0, 1.0
            cont = ax.contourf(slice_disp, levels=np.linspace(0, 1, 3), cmap=cmap_name, vmin=0, vmax=1)
        else:
            local_min = float(np.nanmin(display_field))
            local_max = float(np.nanmax(display_field))
            if np.isclose(local_min, local_max):
                local_max = local_min + 1e-12
            levels = np.linspace(local_min, local_max, 70)
            cont = ax.contourf(slice_disp, levels=levels, cmap=cmap_name, vmin=local_min, vmax=local_max)

            if field_key == "TEMP":
                contour_levels = [lv for lv in [800.0, 1800.0] if local_min <= lv <= local_max]
                if contour_levels:
                    cs = ax.contour(slice_disp, levels=contour_levels, colors="black", linewidths=1.2)
                    ax.clabel(cs, inline=True, fontsize=10, fmt=lambda x: f"{x:.0f} K")

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar = fig_static.colorbar(cont, cax=cax)
        cbar.set_label(plot_unit, fontsize=12, fontweight="bold")
        if is_phase:
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["FCC = 0", "LIQUID = 1"])
        else:
            tickvals = np.linspace(local_min, local_max, 5)
            cbar.set_ticks(tickvals)
            cbar.set_ticklabels([f"{v:.4g}" for v in tickvals])
        cbar.ax.tick_params(labelsize=11)
        fig_static.tight_layout()
        st.pyplot(fig_static)


        display_for_download = display_field
        t_idx_for_download = t_idx_static

    # Downloads
    st.markdown("### Downloads")
    dl1, dl2, dl3 = st.columns(3)
    base_name = f"{field_key}_pred_p{int(p_target)}s{int(v_target_cm_s)}_Co{co_target:.2f}_Cr{cr_target:.2f}_Fe{fe_target:.2f}"
    with dl1:
        st.download_button(
            "Download displayed static .npy",
            data=npy_bytes(display_for_download),
            file_name=f"{base_name}_t{t_idx_for_download}.npy",
            mime="application/octet-stream",
            key=f"download_static_{field_key}",
        )
    with dl2:
        st.download_button(
            "Download full prediction .npy",
            data=npy_bytes(predicted),
            file_name=f"{base_name}_full.npy",
            mime="application/octet-stream",
            key=f"download_full_{field_key}",
        )
    with dl3:
        extra = {
            "field": predicted,
            "P": float(p_target),
            "v_cm_s": float(v_target_cm_s),
            "composition_liquid": target_composition,
            "composition_fcc": target_composition_fcc,
            "ctf_choice": target_ctf_choice,
            "weights": results["combined_weights"],
        }
        if is_phase:
            extra["binary_phase"] = predicted.astype(np.uint8)
            extra["threshold"] = 0.5
        st.download_button(
            "Download .npz with metadata",
            data=npz_bytes(**extra),
            file_name=f"{base_name}.npz",
            mime="application/octet-stream",
            key=f"download_npz_{field_key}",
        )



# ============================================================
# COMBO prediction helpers
# ============================================================
def compute_prediction_for_folder(field_key: str, folder_path: str, use_composition: bool = True, component_index: int = 0):
    """Load a field folder and return a phase-aware attention prediction plus metadata."""
    sources, arrays_raw, warnings, file_count = load_sources_from_folder(folder_path, field_key)
    if len(sources) < 2:
        return {
            "ok": False,
            "field_key": field_key,
            "file_count": file_count,
            "sources": sources,
            "warnings": warnings,
            "message": f"{field_key}: need at least two valid source files.",
        }

    arrays = []
    shapes = []
    for src, arr in zip(sources, arrays_raw):
        try:
            scalar_arr = get_component_view(arr, field_key, component_index)
            arrays.append(scalar_arr)
            shapes.append(scalar_arr.shape)
        except Exception as exc:
            warnings.append(f"{src['file_name']}: {exc}")

    if len(arrays) < 2:
        return {
            "ok": False,
            "field_key": field_key,
            "file_count": file_count,
            "sources": sources,
            "warnings": warnings,
            "message": f"{field_key}: not enough valid scalar fields after conversion.",
        }

    if len(set(shapes)) != 1:
        return {
            "ok": False,
            "field_key": field_key,
            "file_count": file_count,
            "sources": sources,
            "warnings": warnings,
            "message": f"{field_key}: scalar field shapes do not match: {sorted(set(shapes))}",
        }

    model = PhaseAwareAttentionInterpolator(
        sigma_param=sigma_param,
        sigma_comp=sigma_comp,
        num_heads=num_heads,
        d_head=d_head,
        comp_strength=comp_strength,
    )
    results = model.compute_weights(sources, TARGET, use_composition=use_composition)
    weights = results["combined_weights"]
    predicted = np.zeros_like(arrays[0], dtype=np.float64)
    for weight, arr in zip(weights, arrays):
        predicted += weight * arr

    if field_key == "ETALIQ":
        predicted = (predicted >= 0.5).astype(np.float64)

    return {
        "ok": True,
        "field_key": field_key,
        "file_count": file_count,
        "sources": sources,
        "warnings": warnings,
        "prediction": predicted,
        "results": results,
        "shape": predicted.shape,
    }


def transparent_png_bytes(fig) -> io.BytesIO:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    buffer.seek(0)
    return buffer


def plot_combo_velocity(temp_pred: np.ndarray, eta_pred: np.ndarray, vel_pred: np.ndarray,
                        t_idx: int, cmap_name: str, threshold: float):
    """Create the combined velocity plot with LIQUID and temperature contours."""
    temp_slice = np.asarray(temp_pred[t_idx], dtype=float)
    eta_slice = np.asarray(eta_pred[t_idx], dtype=float)
    vel_slice = np.asarray(vel_pred[t_idx], dtype=float)

    liquid_mask = eta_slice >= threshold
    vel_masked = np.where(liquid_mask, vel_slice, 0.0)

    vel_display = np.flipud(vel_masked)
    eta_display = np.flipud(eta_slice)
    temp_display = np.flipud(temp_slice)

    local_min = 0.0
    local_max = float(np.nanmax(vel_display))
    if np.isclose(local_min, local_max):
        local_max = local_min + 1e-12
    levels = np.linspace(local_min, local_max, 80)
    ticks = np.linspace(local_min, local_max, 5)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    cont = ax.contourf(
        vel_display,
        levels=levels,
        cmap=cmap_name,
        vmin=local_min,
        vmax=local_max,
    )

    # LIQUID/FCC interface contour from phase field.
    if float(np.nanmin(eta_display)) <= threshold <= float(np.nanmax(eta_display)):
        cs_liq = ax.contour(
            eta_display,
            levels=[threshold],
            colors="white",
            linewidths=2.0,
            linestyles="solid",
        )
        try:
            cs_liq.collections[0].set_label("LIQUID/FCC interface")
        except Exception:
            pass

    # Temperature contour lines at 800 K and 1800 K.
    tmin = float(np.nanmin(temp_display))
    tmax = float(np.nanmax(temp_display))
    temp_levels = [lv for lv in [800.0, 1800.0] if tmin <= lv <= tmax]
    if temp_levels:
        cs_temp = ax.contour(
            temp_display,
            levels=temp_levels,
            colors="black",
            linewidths=1.5,
            linestyles="--",
        )
        ax.clabel(cs_temp, inline=True, fontsize=10, fmt=lambda x: f"{x:.0f} K")

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(cont, cax=cax)
    cbar.set_label("velocity magnitude", fontsize=12, fontweight="bold")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:.4g}" for v in ticks])
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    stats = {
        "vel_min": local_min,
        "vel_max": local_max,
        "vel_mean_liquid": float(np.nanmean(vel_slice[liquid_mask])) if np.any(liquid_mask) else 0.0,
        "liquid_fraction": float(np.mean(liquid_mask)),
        "temp_min": tmin,
        "temp_max": tmax,
    }
    return fig, vel_masked, stats




def plot_field_animation(field_pred: np.ndarray, cmap_name: str, title: str, colorbar_title: str,
                         hover_label: str, zmin: Optional[float] = None, zmax: Optional[float] = None,
                         threshold_contour: Optional[float] = None) -> go.Figure:
    """Create a standalone Plotly heatmap animation for one predicted field."""
    Nt = field_pred.shape[0]
    colorscale = mpl_to_plotly_colorscale(cmap_name)

    global_min = float(np.nanmin(field_pred)) if zmin is None else float(zmin)
    global_max = float(np.nanmax(field_pred)) if zmax is None else float(zmax)
    if np.isclose(global_min, global_max):
        global_max = global_min + 1e-12

    ticks = np.linspace(global_min, global_max, 5)

    def frame_data(k: int):
        display_slice = np.flipud(np.asarray(field_pred[k], dtype=float))
        heat = go.Heatmap(
            z=display_slice,
            colorscale=colorscale,
            zmin=global_min,
            zmax=global_max,
            colorbar=dict(
                title=colorbar_title,
                tickmode="array",
                tickvals=ticks.tolist(),
                ticktext=[f"{v:.4g}" for v in ticks],
            ),
            hovertemplate=f"x=%{{x}}<br>y=%{{y}}<br>{hover_label}=%{{z:.4g}}<extra></extra>",
            name=title,
        )
        traces = [heat]

        if threshold_contour is not None:
            if float(np.nanmin(display_slice)) <= threshold_contour <= float(np.nanmax(display_slice)):
                traces.append(
                    go.Contour(
                        z=display_slice,
                        contours=dict(
                            coloring="lines",
                            showlabels=False,
                            start=float(threshold_contour),
                            end=float(threshold_contour),
                            size=1.0,
                        ),
                        line=dict(color="white", width=3),
                        showscale=False,
                        hoverinfo="skip",
                        name="LIQUID/FCC interface",
                    )
                )
        return traces

    frames = [go.Frame(data=frame_data(k), name=f"t{k}") for k in range(Nt)]
    fig = go.Figure(data=frame_data(0), frames=frames)
    fig.update_layout(
        title=title,
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title="", scaleanchor="x", scaleratio=1),
        updatemenus=[{
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
            "type": "buttons",
            "x": 0.05,
            "y": 0,
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {"method": "animate", "args": [[f"t{k}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], "label": str(k)}
                for k in range(Nt)
            ],
            "x": 0.1,
            "len": 0.85,
            "y": 0,
            "currentvalue": {"prefix": "Time index: "},
        }],
        margin=dict(l=0, r=0, t=55, b=0),
        height=720,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def plot_combo_animation(temp_pred: np.ndarray, eta_pred: np.ndarray, vel_pred: np.ndarray,
                         cmap_name: str, threshold: float) -> go.Figure:
    """Create Plotly animation for COMBO: masked velocity + LIQUID and temperature contours."""
    Nt = vel_pred.shape[0]
    colorscale = mpl_to_plotly_colorscale(cmap_name)

    def make_frame_data(k: int):
        temp_slice = np.asarray(temp_pred[k], dtype=float)
        eta_slice = np.asarray(eta_pred[k], dtype=float)
        vel_slice = np.asarray(vel_pred[k], dtype=float)

        liquid_mask = eta_slice >= threshold
        vel_masked = np.where(liquid_mask, vel_slice, 0.0)

        vel_display = np.flipud(vel_masked)
        eta_display = np.flipud(eta_slice)
        temp_display = np.flipud(temp_slice)

        local_min = 0.0
        local_max = float(np.nanmax(vel_display))
        if np.isclose(local_min, local_max):
            local_max = local_min + 1e-12
        vel_ticks = np.linspace(local_min, local_max, 5)

        heat = go.Heatmap(
            z=vel_display,
            colorscale=colorscale,
            zmin=local_min,
            zmax=local_max,
            colorbar=dict(
                title="velocity magnitude",
                tickmode="array",
                tickvals=vel_ticks.tolist(),
                ticktext=[f"{v:.4g}" for v in vel_ticks],
            ),
            hovertemplate="x=%{x}<br>y=%{y}<br>velocity=%{z:.4g}<extra></extra>",
            name="Masked velocity",
        )

        # LIQUID/FCC interface contour at the selected threshold.
        liq_contour = go.Contour(
            z=eta_display,
            contours=dict(
                coloring="lines",
                showlabels=False,
                start=float(threshold),
                end=float(threshold),
                size=1.0,
            ),
            line=dict(color="white", width=3),
            showscale=False,
            hoverinfo="skip",
            name="LIQUID/FCC interface",
        )

        # Temperature contours at 800 K and 1800 K.
        tmin = float(np.nanmin(temp_display))
        tmax = float(np.nanmax(temp_display))
        temp_levels = [lv for lv in [800.0, 1800.0] if tmin <= lv <= tmax]
        if len(temp_levels) == 2:
            start, end, size = 800.0, 1800.0, 1000.0
            opacity = 1.0
            z_temp = temp_display
        elif len(temp_levels) == 1:
            start, end, size = temp_levels[0], temp_levels[0], 1.0
            opacity = 1.0
            z_temp = temp_display
        else:
            start, end, size = 0.0, 0.0, 1.0
            opacity = 0.0
            z_temp = np.zeros_like(temp_display)

        temp_contour = go.Contour(
            z=z_temp,
            contours=dict(
                coloring="lines",
                showlabels=True,
                labelfont=dict(size=12, color="black"),
                start=start,
                end=end,
                size=size,
            ),
            line=dict(color="black", width=2),
            showscale=False,
            opacity=opacity,
            hoverinfo="skip",
            name="Temperature contours",
        )

        return [heat, liq_contour, temp_contour]

    frames = [go.Frame(data=make_frame_data(k), name=f"t{k}") for k in range(Nt)]
    fig = go.Figure(data=make_frame_data(0), frames=frames)
    fig.update_layout(
        title="COMBO evolution: velocity masked by LIQUID phase with phase/temperature contours",
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title="", scaleanchor="x", scaleratio=1),
        updatemenus=[{
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
            "type": "buttons",
            "x": 0.05,
            "y": 0,
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {"method": "animate", "args": [[f"t{k}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], "label": str(k)}
                for k in range(Nt)
            ],
            "x": 0.1,
            "len": 0.85,
            "y": 0,
            "currentvalue": {"prefix": "Time index: "},
        }],
        margin=dict(l=0, r=0, t=55, b=0),
        height=720,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def run_combo_tab():
    st.subheader("COMBO Prediction")
    st.markdown(
        "This tab predicts TEMP, ETALIQ and VEL together. "
        "It visualizes temperature distribution, ETALIQ phase evolution, and velocity profile as separate Plotly animations. "
        "For velocity, values are kept only inside the predicted LIQUID region; the FCC/solid region is set to 0, with LIQUID/FCC boundary and 800 K / 1800 K temperature contours overlaid."
    )

    c1, c2, c3 = st.columns(3)
    combo_folders = {}
    for col, key, label in [(c1, "TEMP", "Temperature"), (c2, "ETALIQ", "ETALIQ"), (c3, "VEL", "Velocity")]:
        folder_key = f"combo_folder_{key}"
        if folder_key not in st.session_state:
            st.session_state[folder_key] = st.session_state.get(f"folder_{key}", DEFAULT_DIRS[key])
        with col:
            st.text_input(f"{label} folder", key=folder_key)
            if st.button(f"Browse {key}", key=f"combo_browse_{key}"):
                selected_folder = choose_folder_dialog(st.session_state[folder_key])
                if selected_folder:
                    st.session_state[folder_key] = selected_folder
                    st.rerun()
            combo_folders[key] = st.session_state[folder_key]

    use_comp_combo = st.checkbox("Use composition similarity in COMBO weighting", value=True, key="use_comp_COMBO")

    if st.button("Run COMBO prediction", type="primary", disabled=not target_valid, key="run_COMBO"):
        with st.spinner("Predicting TEMP, ETALIQ and VEL for COMBO plot..."):
            combo = {
                "TEMP": compute_prediction_for_folder("TEMP", combo_folders["TEMP"], use_composition=use_comp_combo),
                "ETALIQ": compute_prediction_for_folder("ETALIQ", combo_folders["ETALIQ"], use_composition=use_comp_combo),
                "VEL": compute_prediction_for_folder("VEL", combo_folders["VEL"], use_composition=use_comp_combo),
            }
            st.session_state["combo_predictions"] = combo

    combo = st.session_state.get("combo_predictions")
    if combo is None:
        st.info("Select the TEMP, ETALIQ and VEL folders, then run COMBO prediction.")
        return

    # Loading summaries and warnings.
    cols = st.columns(3)
    for col, key, label in zip(cols, ["TEMP", "ETALIQ", "VEL"], ["Temperature", "ETALIQ", "Velocity"]):
        res = combo.get(key, {})
        with col:
            st.info(f"Found {res.get('file_count', 0)} .npy files in {label} ({os.path.basename(combo_folders[key])}/).")
            if res.get("sources"):
                st.markdown(f"**📋 Loaded {len(res['sources'])} Valid Source Simulations**")
                st.dataframe(sources_dataframe(res["sources"]), width='stretch', height=220)
            for w in res.get("warnings", [])[:3]:
                st.warning(w)
            if len(res.get("warnings", [])) > 3:
                st.warning(f"Additional warnings hidden: {len(res['warnings']) - 3}")
            if not res.get("ok", False):
                st.error(res.get("message", f"{key} prediction failed."))

    if not all(combo.get(k, {}).get("ok", False) for k in ["TEMP", "ETALIQ", "VEL"]):
        return

    temp_pred = combo["TEMP"]["prediction"]
    eta_pred = (combo["ETALIQ"]["prediction"] >= 0.5).astype(np.float64)
    vel_pred = combo["VEL"]["prediction"]

    if temp_pred.shape != eta_pred.shape or temp_pred.shape != vel_pred.shape:
        st.error(f"Predicted shapes must match. TEMP={temp_pred.shape}, ETALIQ={eta_pred.shape}, VEL={vel_pred.shape}")
        return

    Nt, Ny, Nx = vel_pred.shape

    st.markdown("### COMBO synchronized Plotly animation")
    st.caption(
        "Use the Plotly time-index slider at the bottom. It updates temperature, ETALIQ phase, and masked velocity together without refreshing the Streamlit page. ETALIQ is binarized with fixed threshold 0.50."
    )

    from plotly.subplots import make_subplots

    eta_global_min, eta_global_max = 0.0, 1.0
    eta_ticks = np.array([0.0, 1.0])

    liquid_masks = eta_pred >= 0.5
    vel_masked_all = np.where(liquid_masks, vel_pred, 0.0)

    def frame_min_max_ticks(arr: np.ndarray, n_ticks: int = 5):
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if np.isclose(vmin, vmax):
            pad = max(abs(vmax) * 1e-6, 1e-12)
            vmin -= pad
            vmax += pad
        ticks = np.linspace(vmin, vmax, n_ticks)
        return vmin, vmax, ticks

    def combo_frame_traces(k: int):
        temp_slice = np.asarray(temp_pred[k], dtype=float)
        eta_slice = np.asarray(eta_pred[k], dtype=float)
        vel_slice = np.asarray(vel_masked_all[k], dtype=float)

        temp_display = np.flipud(temp_slice)
        eta_display = np.flipud(eta_slice)
        vel_display = np.flipud(vel_slice)

        temp_frame_min, temp_frame_max, temp_ticks = frame_min_max_ticks(temp_display)
        vel_frame_min, vel_frame_max, vel_ticks = frame_min_max_ticks(vel_display)

        traces = [
            go.Heatmap(
                z=temp_display,
                colorscale=mpl_to_plotly_colorscale(temp_cmap),
                zmin=temp_frame_min,
                zmax=temp_frame_max,
                coloraxis=None,
                colorbar=dict(
                    title="Temperature (K)",
                    x=1.02,
                    y=0.855,
                    len=0.25,
                    tickmode="array",
                    tickvals=temp_ticks.tolist(),
                    ticktext=[f"{v:.4g}" for v in temp_ticks],
                ),
                hovertemplate="x=%{x}<br>y=%{y}<br>T=%{z:.4g} K<extra></extra>",
                name="Temperature",
                showscale=True,
            ),
            go.Heatmap(
                z=eta_display,
                colorscale=mpl_to_plotly_colorscale(phase_cmap),
                zmin=eta_global_min,
                zmax=eta_global_max,
                colorbar=dict(
                    title="ETALIQ",
                    x=1.02,
                    y=0.505,
                    len=0.25,
                    tickmode="array",
                    tickvals=eta_ticks.tolist(),
                    ticktext=["FCC = 0", "LIQUID = 1"],
                ),
                hovertemplate="x=%{x}<br>y=%{y}<br>ETALIQ=%{z:.4g}<extra></extra>",
                name="ETALIQ",
                showscale=True,
            ),
            go.Contour(
                z=eta_display,
                contours=dict(
                    coloring="lines",
                    showlabels=False,
                    start=0.5,
                    end=0.5,
                    size=1.0,
                ),
                line=dict(color="white", width=3),
                showscale=False,
                hoverinfo="skip",
                name="LIQUID/FCC interface",
            ),
            go.Heatmap(
                z=vel_display,
                colorscale=mpl_to_plotly_colorscale(vel_cmap),
                zmin=vel_frame_min,
                zmax=vel_frame_max,
                colorbar=dict(
                    title="Velocity",
                    x=1.02,
                    y=0.155,
                    len=0.25,
                    tickmode="array",
                    tickvals=vel_ticks.tolist(),
                    ticktext=[f"{v:.4g}" for v in vel_ticks],
                ),
                hovertemplate="x=%{x}<br>y=%{y}<br>velocity=%{z:.4g}<extra></extra>",
                name="Masked velocity",
                showscale=True,
            ),
            go.Contour(
                z=eta_display,
                contours=dict(
                    coloring="lines",
                    showlabels=False,
                    start=0.5,
                    end=0.5,
                    size=1.0,
                ),
                line=dict(color="white", width=3),
                showscale=False,
                hoverinfo="skip",
                name="LIQUID/FCC interface",
            ),
        ]

        tmin = float(np.nanmin(temp_display))
        tmax = float(np.nanmax(temp_display))
        temp_levels = [lv for lv in [800.0, 1800.0] if tmin <= lv <= tmax]
        if len(temp_levels) == 2:
            z_temp, start_t, end_t, size_t, opacity_t = temp_display, 800.0, 1800.0, 1000.0, 1.0
        elif len(temp_levels) == 1:
            z_temp, start_t, end_t, size_t, opacity_t = temp_display, temp_levels[0], temp_levels[0], 1.0, 1.0
        else:
            z_temp, start_t, end_t, size_t, opacity_t = np.zeros_like(temp_display), 0.0, 0.0, 1.0, 0.0

        traces.append(
            go.Contour(
                z=z_temp,
                contours=dict(
                    coloring="lines",
                    showlabels=True,
                    labelfont=dict(size=12, color="black"),
                    start=start_t,
                    end=end_t,
                    size=size_t,
                ),
                line=dict(color="black", width=2),
                showscale=False,
                opacity=opacity_t,
                hoverinfo="skip",
                name="Temperature contours",
            )
        )
        return traces

    combo_fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.065,
        subplot_titles=(
            "1. Temperature distribution",
            "2. ETALIQ phase evolution",
            "3. Velocity profile masked by LIQUID phase",
        ),
    )

    initial_traces = combo_frame_traces(0)
    combo_fig.add_trace(initial_traces[0], row=1, col=1)
    combo_fig.add_trace(initial_traces[1], row=2, col=1)
    combo_fig.add_trace(initial_traces[2], row=2, col=1)
    combo_fig.add_trace(initial_traces[3], row=3, col=1)
    combo_fig.add_trace(initial_traces[4], row=3, col=1)
    combo_fig.add_trace(initial_traces[5], row=3, col=1)

    frame_trace_indexes = [0, 1, 2, 3, 4, 5]
    combo_fig.frames = [
        go.Frame(
            data=combo_frame_traces(k),
            traces=frame_trace_indexes,
            name=f"t{k}",
            layout=go.Layout(title_text=f"COMBO Prediction | Time index: {k}"),
        )
        for k in range(Nt)
    ]

    steps = [
        {
            "method": "animate",
            "args": [
                [f"t{k}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
            "label": str(k),
        }
        for k in range(Nt)
    ]

    combo_fig.update_layout(
        title="COMBO Prediction | Time index: 0",
        height=1850,
        margin=dict(l=25, r=105, t=90, b=130),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 120, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.055,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": steps,
                "x": 0.12,
                "y": -0.06,
                "len": 0.86,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 35, "b": 10},
                "currentvalue": {"prefix": "Time index: ", "font": {"size": 18}},
            }
        ],
    )

    for axis_name in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
        combo_fig.layout[axis_name].showticklabels = False
        combo_fig.layout[axis_name].title = ""

    combo_fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    combo_fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=2, col=1)
    combo_fig.update_yaxes(scaleanchor="x3", scaleratio=1, row=3, col=1)

    st.plotly_chart(combo_fig, width='stretch', key="combo_three_panel_single_plotly_slider")

    vel_values_liquid = vel_pred[liquid_masks]
    mean_liquid = float(np.nanmean(vel_values_liquid)) if vel_values_liquid.size else 0.0
    st.write(
        "Colorbar limits for TEMP and masked VEL are updated from the current Plotly time-index frame. "
        f"Mean velocity over LIQUID pixels across all time steps = {mean_liquid:.4g}."
    )

# ============================================================
# Tabs
# ============================================================
tab_temp, tab_eta, tab_vel, tab_combo, tab_comp = st.tabs([
    "TEMP prediction",
    "ETALIQ prediction",
    "VEL prediction",
    "COMBO Prediction",
    "COMP prediction",
])

with tab_temp:
    run_field_tab("TEMP", "Temperature", "K", DEFAULT_DIRS["TEMP"], temp_cmap, is_phase=False)

with tab_eta:
    st.info("ETALIQ source fields are assumed to be LIQUID = 1 and FCC = 0. No LIQ = 0.5 contour line is drawn.")
    run_field_tab("ETALIQ", "ETALIQ phase", "phase", DEFAULT_DIRS["ETALIQ"], phase_cmap, is_phase=True)

with tab_vel:
    st.info("Velocity data are loaded from the VEL folder. If files contain vector velocity, magnitude is calculated automatically.")
    run_field_tab("VEL", "Velocity", "velocity magnitude", DEFAULT_DIRS["VEL"], vel_cmap, is_phase=False)

with tab_combo:
    run_combo_tab()

with tab_comp:
    run_field_tab("COMP", "Composition", "mole fraction", DEFAULT_DIRS["COMP"], comp_cmap, is_phase=False)

# ============================================================
# Footer theory note
# ============================================================
with st.expander("Theory/calculation used in this version", expanded=False):
    st.markdown(
        """
The calculation follows the phase-aware attention idea from the colleague version:

1. Source filenames are parsed as `p<P>s<v>cTF<k>.npy`.
2. Process parameters `(P, v)` are min-max normalized.
3. Query/key projections are computed with a multi-head attention layer.
4. A Gaussian locality term favors nearby `(P, v)` sources.
5. A Gaussian composition-similarity term compares the target phase-specific `[Co, Cr, Fe, Ni]` vectors against the source `cTF` LIQUID and FCC composition vectors.
6. The final normalized hybrid weight is used to interpolate the full spatiotemporal field.

The plots are kept as direct static/animated field visualizations with manual colormap selection, temperature contours at 800 K and 1800 K, and `.npy/.npz` downloads.
"""
    )
