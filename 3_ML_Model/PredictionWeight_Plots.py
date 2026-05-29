import os
import re
import io
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
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

# cTF table used only to map existing filenames to source initial compositions.
# The target composition is selected directly with Co/Cr/Fe sliders in the sidebar.
CTF_TABLE: Dict[int, Dict[str, List[float]]] = {
    0: {"liquid": [0.35, 0.13, 0.15, 0.37], "fcc": [0.28, 0.18, 0.22, 0.32]},
    1: {"liquid": [0.32, 0.15, 0.17, 0.36], "fcc": [0.25, 0.21, 0.25, 0.29]},
    2: {"liquid": [0.30, 0.19, 0.11, 0.40], "fcc": [0.25, 0.23, 0.20, 0.32]},
    3: {"liquid": [0.33, 0.11, 0.23, 0.33], "fcc": [0.31, 0.15, 0.27, 0.27]},
    4: {"liquid": [0.38, 0.17, 0.19, 0.26], "fcc": [0.30, 0.22, 0.26, 0.22]},
}

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
initial composition set. The target composition is selected directly as Co, Cr and Fe;
Ni is inferred from mass conservation.
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
        cL = s["composition_liquid"]
        rows.append({
            "File": s["file_name"],
            "P (W)": s["P"],
            "v (cm/s)": s["v_cm_s"],
            "cTF": s["TF_idx"],
            "Co": cL[0],
            "Cr": cL[1],
            "Fe": cL[2],
            "Ni": cL[3],
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

def plotly_figure_to_transparent_png_bytes(fig: go.Figure, scale: int = 3) -> Optional[bytes]:
    """Export a Plotly figure as a transparent PNG.

    Returns None if static image export is not available in the runtime.
    """
    try:
        export_fig = go.Figure(fig)
        export_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        if hasattr(export_fig.layout, "legend") and export_fig.layout.legend is not None:
            export_fig.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)"))

        if hasattr(export_fig.layout, "scene") and export_fig.layout.scene is not None:
            export_fig.update_scenes(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            )

        return export_fig.to_image(format="png", scale=scale)
    except Exception:
        return None


def add_transparent_png_download(fig: go.Figure, filename: str, label: str, key: str, scale: int = 3):
    png_bytes = plotly_figure_to_transparent_png_bytes(fig, scale=scale)
    if png_bytes is None:
        st.info("Static PNG export requires Plotly image export support (for example, kaleido) in the runtime environment.")
        return
    st.download_button(
        label=label,
        data=png_bytes,
        file_name=filename,
        mime="image/png",
        key=key,
    )


def render_plotly_with_live_transparent_download(
    fig: go.Figure,
    filename: str,
    button_label: str,
    key: str,
    height: int = 820,
    scale: int = 3,
):
    """Render a Plotly figure in an HTML component and download the *current* displayed view.

    The download button uses Plotly.downloadImage on the client-side graph div, so
    camera orientation / zoom / pan / current slider-driven appearance are preserved.
    """
    safe_key = re.sub(r"[^A-Za-z0-9_-]", "_", str(key))
    div_id = f"plotly_live_{safe_key}"
    button_id = f"download_btn_{safe_key}"
    filename_stem = os.path.splitext(filename)[0]

    config = {
        "responsive": True,
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename_stem, "scale": scale},
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }

    plot_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config=config,
        div_id=div_id,
    )

    html = f"""
    <div style="width:100%;">
        {plot_html}
        <div style="margin-top:10px;margin-bottom:4px;">
            <button id="{button_id}"
                style="background-color:#f0f2f6;color:#111;border:1px solid rgba(49,51,63,0.2);
                       padding:0.45rem 0.9rem;border-radius:0.5rem;cursor:pointer;
                       font-size:0.95rem;font-weight:600;">
                {button_label}
            </button>
        </div>
    </div>
    <script>
    (function() {{
        const gd = document.getElementById({json.dumps(div_id)});
        const btn = document.getElementById({json.dumps(button_id)});
        if (!gd || !btn || typeof Plotly === 'undefined') return;

        btn.addEventListener('click', async function() {{
            const restore = {{
                paper_bgcolor: gd.layout.paper_bgcolor || null,
                plot_bgcolor: gd.layout.plot_bgcolor || null,
                legend_bgcolor: (gd.layout.legend && gd.layout.legend.bgcolor) ? gd.layout.legend.bgcolor : null,
                scene_x_bg: (gd.layout.scene && gd.layout.scene.xaxis && gd.layout.scene.xaxis.backgroundcolor) ? gd.layout.scene.xaxis.backgroundcolor : null,
                scene_y_bg: (gd.layout.scene && gd.layout.scene.yaxis && gd.layout.scene.yaxis.backgroundcolor) ? gd.layout.scene.yaxis.backgroundcolor : null,
                scene_z_bg: (gd.layout.scene && gd.layout.scene.zaxis && gd.layout.scene.zaxis.backgroundcolor) ? gd.layout.scene.zaxis.backgroundcolor : null
            }};

            const exportUpdates = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                'legend.bgcolor': 'rgba(0,0,0,0)'
            }};

            if (gd.layout.scene) {{
                exportUpdates['scene.xaxis.backgroundcolor'] = 'rgba(0,0,0,0)';
                exportUpdates['scene.yaxis.backgroundcolor'] = 'rgba(0,0,0,0)';
                exportUpdates['scene.zaxis.backgroundcolor'] = 'rgba(0,0,0,0)';
            }}

            const restoreUpdates = {{
                paper_bgcolor: restore.paper_bgcolor,
                plot_bgcolor: restore.plot_bgcolor,
                'legend.bgcolor': restore.legend_bgcolor
            }};

            if (gd.layout.scene) {{
                restoreUpdates['scene.xaxis.backgroundcolor'] = restore.scene_x_bg;
                restoreUpdates['scene.yaxis.backgroundcolor'] = restore.scene_y_bg;
                restoreUpdates['scene.zaxis.backgroundcolor'] = restore.scene_z_bg;
            }}

            try {{
                await Plotly.relayout(gd, exportUpdates);
                await Plotly.downloadImage(gd, {{
                    format: 'png',
                    filename: {json.dumps(filename_stem)},
                    scale: {int(scale)}
                }});
            }} finally {{
                await Plotly.relayout(gd, restoreUpdates);
            }}
        }});
    }})();
    </script>
    """
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


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
        diff2 = float(np.sum((np.asarray(source_comp) - np.asarray(target_comp)) ** 2))
        sim = np.exp(-diff2 / (2 * self.sigma_comp ** 2))
        return float(sim)

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
            sims = np.array([
                self.composition_similarity(s["composition_liquid"], target["composition"])
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
    p_target = st.number_input("Power, P (W)", min_value=300.0, max_value=800.0, value=435.0, step=10.0)
    v_target_cm_s = st.number_input("Scan speed, v (cm/s)", min_value=40.0, max_value=80.0, value=55.0, step=5.0)

    co_target = st.slider("Co", 0.01, 0.40, 0.35, 0.01)
    cr_target = st.slider("Cr", 0.01, 0.40, 0.13, 0.01)
    fe_target = st.slider("Fe", 0.01, 0.40, 0.15, 0.01)
    ni_target = round(1.0 - co_target - cr_target - fe_target, 4)

    if ni_target < 0.01:
        st.error(f"Invalid composition: Ni = {ni_target:.3f}. Reduce Co, Cr or Fe so Ni ≥ 0.01.")
        target_valid = False
    elif ni_target > 0.97:
        st.error(f"Invalid composition: Ni = {ni_target:.3f}. Increase Co, Cr or Fe.")
        target_valid = False
    else:
        st.success(f"Ni = 1 - Co - Cr - Fe = {ni_target:.2f}")
        target_valid = True

    target_composition = np.array([co_target, cr_target, fe_target, ni_target], dtype=float)

    st.header("Attention model")
    sigma_param = st.slider("Gaussian locality σ for P-v", 0.05, 0.50, 0.20, 0.01)
    sigma_comp = st.slider("Composition similarity σ", 0.01, 0.25, 0.05, 0.01)
    comp_strength = st.slider("Composition influence strength", 0.0, 2.0, 1.0, 0.1)
    num_heads = st.slider("Attention heads", 1, 8, 4, 1)
    d_head = st.slider("Dimension per head", 4, 16, 8, 1)
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
    "composition": target_composition,
}


def target_download_prefix() -> str:
    """Return compact target-condition prefix such as p435s55."""
    p_val = int(round(float(p_target)))
    s_val = int(round(float(v_target_cm_s)))
    return f"p{p_val}s{s_val}"


# ============================================================
# Source-to-target prediction weight visualization
# ============================================================

def plot_source_target_weight_visualization(
    cached_sources: List[Dict],
    results: Dict[str, np.ndarray],
    field_key: str,
):
    """Visualize how source simulations contribute to the current target prediction."""
    if not cached_sources or results is None:
        return

    st.markdown("### Source-to-target prediction weight visualization")

    src_p = np.array([s["P"] for s in cached_sources], dtype=float)
    src_v = np.array([s["v_cm_s"] for s in cached_sources], dtype=float)
    src_tf = np.array([s["TF_idx"] for s in cached_sources], dtype=int)
    src_comp = np.array([s["composition_liquid"] for s in cached_sources], dtype=float)
    tgt_comp = np.array(target_composition, dtype=float)

    attn_w = np.array(results["attention_weights_proc"], dtype=float)
    proc_w = np.array(results["spatial_weights_proc"], dtype=float)
    comp_w = np.array(results["composition_weights"], dtype=float)
    final_w = np.array(results["combined_weights"], dtype=float)

    comp_dist = np.linalg.norm(src_comp - tgt_comp[None, :], axis=1)
    proc_dist_norm = np.linalg.norm(results["norm_sources_proc"] - results["norm_target_proc"][None, :], axis=1)

    max_w = float(np.nanmax(final_w)) if len(final_w) else 1.0
    if max_w <= 0 or not np.isfinite(max_w):
        max_w = 1.0

    marker_size = 14 + 58 * (final_w / max_w)
    line_width = 1.0 + 10.0 * (final_w / max_w)

    source_labels = [
        f"p{int(s['P'])}s{int(s['v_cm_s'])}cTF{int(s['TF_idx'])}"
        for s in cached_sources
    ]

    weight_df = pd.DataFrame({
        "Source": source_labels,
        "File": [s["file_name"] for s in cached_sources],
        "P (W)": src_p,
        "v (cm/s)": src_v,
        "cTF": src_tf,
        "Co": src_comp[:, 0],
        "Cr": src_comp[:, 1],
        "Fe": src_comp[:, 2],
        "Ni": src_comp[:, 3],
        "Composition distance": comp_dist,
        "Normalized P-v distance": proc_dist_norm,
        "Attention": attn_w,
        "P-v Gaussian": proc_w,
        "Composition": comp_w,
        "Final weight": final_w,
    })

    weight_df_sorted = weight_df.sort_values("Final weight", ascending=False).reset_index(drop=True)
    top_source = weight_df_sorted.iloc[0]
    st.info(
        f"Dominant source for this target: `{top_source['Source']}` "
        f"with final weight = {top_source['Final weight']:.5f}."
    )

    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "3D process-composition map",
        "Chord contribution map",
        "Weight decomposition",
        "Query-Key heatmaps",
    ])

    # ------------------------------------------------------------
    # 1. 3D Power-Speed-Composition-distance map
    # ------------------------------------------------------------
    with viz_tab1:
        axis_tick_label_padding = st.slider(
            "3D axis tick/label padding",
            min_value=0,
            max_value=40,
            value=16,
            step=1,
            key=f"axis_tick_label_padding_3d_{field_key}",
        )
        axis_title_pad = " " * int(axis_tick_label_padding // 2)

        fig_3d = go.Figure()

        for i in range(len(cached_sources)):
            fig_3d.add_trace(go.Scatter3d(
                x=[src_p[i], float(p_target)],
                y=[src_v[i], float(v_target_cm_s)],
                z=[comp_dist[i], 0.0],
                mode="lines",
                line=dict(
                    width=float(line_width[i]),
                    color="rgba(0, 0, 0, 1.0)",
                ),
                hoverinfo="skip",
                showlegend=False,
            ))

        fig_3d.add_trace(go.Scatter3d(
            x=src_p,
            y=src_v,
            z=comp_dist,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=final_w,
                colorscale="Plasma",
                showscale=True,
                opacity=0.96,
                colorbar=dict(
                    title=dict(text="<b>Final<br>weight</b>", font=dict(size=24, color="black")),
                    thickness=54,
                    len=0.78,
                    tickfont=dict(size=20, color="black"),
                    outlinewidth=2.5,
                    outlinecolor="black",
                ),
                line=dict(width=2.4, color="black"),
            ),
            text=source_labels,
            customdata=np.column_stack([
                src_tf,
                src_comp[:, 0],
                src_comp[:, 1],
                src_comp[:, 2],
                src_comp[:, 3],
                attn_w,
                proc_w,
                comp_w,
                final_w,
                proc_dist_norm,
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Power: %{x:.0f} W<br>"
                "Scan speed: %{y:.0f} cm/s<br>"
                "Composition distance: %{z:.5f}<br>"
                "Normalized P-v distance: %{customdata[9]:.5f}<br>"
                "cTF: %{customdata[0]}<br>"
                "Co: %{customdata[1]:.3f}<br>"
                "Cr: %{customdata[2]:.3f}<br>"
                "Fe: %{customdata[3]:.3f}<br>"
                "Ni: %{customdata[4]:.3f}<br>"
                "Attention: %{customdata[5]:.5f}<br>"
                "P-v Gaussian: %{customdata[6]:.5f}<br>"
                "Composition: %{customdata[7]:.5f}<br>"
                "<b>Final weight: %{customdata[8]:.5f}</b>"
                "<extra></extra>"
            ),
            name="Source simulations",
        ))

        # Plotly Scatter3d does not support marker symbol="star".
        # A compact star is drawn manually with short 3D line rays plus a small center marker.
        target_hovertemplate = (
            "<b>Target prediction</b><br>"
            f"Power: {float(p_target):.2f} W<br>"
            f"Scan speed: {float(v_target_cm_s):.2f} cm/s<br>"
            f"Co: {tgt_comp[0]:.3f}<br>"
            f"Cr: {tgt_comp[1]:.3f}<br>"
            f"Fe: {tgt_comp[2]:.3f}<br>"
            f"Ni: {tgt_comp[3]:.3f}"
            "<extra></extra>"
        )

        tx = float(p_target)
        ty = float(v_target_cm_s)
        tz = 0.0
        p_span = max(float(np.max(src_p) - np.min(src_p)), 1.0)
        v_span = max(float(np.max(src_v) - np.min(src_v)), 1.0)
        z_span = max(float(np.max(comp_dist) - np.min(comp_dist)), 1e-3)

        # Small, publication-friendly star size. The y/z lengths are scaled to axis span
        # so the target does not look oversized when the axes have different units.
        star_dx = 0.025 * p_span
        star_dy = 0.025 * v_span
        star_dz = 0.025 * z_span
        star_segments = [
            ((-star_dx, 0.0, 0.0), (star_dx, 0.0, 0.0)),
            ((0.0, -star_dy, 0.0), (0.0, star_dy, 0.0)),
            ((-0.70 * star_dx, -0.70 * star_dy, 0.0), (0.70 * star_dx, 0.70 * star_dy, 0.0)),
            ((-0.70 * star_dx, 0.70 * star_dy, 0.0), (0.70 * star_dx, -0.70 * star_dy, 0.0)),
            ((0.0, 0.0, -star_dz), (0.0, 0.0, star_dz)),
        ]

        first_star_segment = True
        for (x0, y0, z0), (x1, y1, z1) in star_segments:
            fig_3d.add_trace(go.Scatter3d(
                x=[tx + x0, tx + x1],
                y=[ty + y0, ty + y1],
                z=[tz + z0, tz + z1],
                mode="lines",
                line=dict(color="red", width=8),
                hovertemplate=target_hovertemplate,
                name="Target" if first_star_segment else "Target star overlay",
                showlegend=first_star_segment,
            ))
            first_star_segment = False

        fig_3d.add_trace(go.Scatter3d(
            x=[tx],
            y=[ty],
            z=[tz],
            mode="markers",
            marker=dict(
                size=7,
                symbol="circle",
                color="red",
                line=dict(width=1.8, color="black"),
            ),
            hovertemplate=target_hovertemplate,
            name="Target center",
            showlegend=False,
        ))

        axis_title_font_3d = dict(size=22, color="black", family="Arial Black")
        axis_common = dict(
            tickfont=dict(size=18, color="black", family="Arial Black"),
            ticklen=int(axis_tick_label_padding),
            tickwidth=3,
            tickprefix=" " * int(axis_tick_label_padding // 6),
            ticksuffix=" " * int(axis_tick_label_padding // 6),
            showgrid=True,
            gridcolor="rgba(20,20,20,0.62)",
            gridwidth=3,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=3,
            linecolor="black",
            linewidth=4,
            showbackground=True,
            backgroundcolor="rgba(245,245,245,0.78)",
        )

        fig_3d.update_layout(
            title=dict(
                text=(
                    "<b>3D source contribution map for target prediction</b>"
                    "<br><sup><b>z-axis = Euclidean composition distance; marker size and color = final hybrid weight</b></sup>"
                ),
                font=dict(size=28, color="black", family="Arial Black"),
                x=0.5,
                xanchor="center",
            ),
            scene=dict(
                xaxis=dict(title=dict(text=f"<b>{axis_title_pad}Power, P (W){axis_title_pad}</b>", font=axis_title_font_3d), **axis_common),
                yaxis=dict(title=dict(text=f"<b>{axis_title_pad}Scan speed, v (cm/s){axis_title_pad}</b>", font=axis_title_font_3d), **axis_common),
                zaxis=dict(title=dict(text=f"<b>{axis_title_pad}Composition distance{axis_title_pad}</b>", font=axis_title_font_3d), **axis_common),
                camera=dict(eye=dict(x=1.65, y=1.75, z=1.25)),
                aspectmode="cube",
            ),
            height=930,
            margin=dict(l=0, r=0, t=105, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                font=dict(size=17, color="black", family="Arial Black"),
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=18, color="black", family="Arial Black"),
        )
        render_plotly_with_live_transparent_download(
            fig_3d,
            filename=f"{target_download_prefix()}_{field_key.lower()}_3d_process_composition_map.png",
            button_label="Download transparent PNG",
            key=f"source_target_3d_{field_key}",
            height=980,
            scale=3,
        )

    # ------------------------------------------------------------
    # 2. Chord-style contribution diagram
    # ------------------------------------------------------------
    with viz_tab2:
        control_col1, control_col2, control_col3 = st.columns([1.0, 1.0, 1.0])
        with control_col1:
            top_n_chord = st.slider(
                "Number of strongest source simulations",
                min_value=1,
                max_value=len(weight_df_sorted),
                value=min(24, len(weight_df_sorted)),
                step=1,
                key=f"top_n_chord_{field_key}",
            )
        with control_col2:
            weight_cmap = st.selectbox(
                "Weight colormap",
                CMAP_OPTIONS,
                index=CMAP_OPTIONS.index("plasma") if "plasma" in CMAP_OPTIONS else 0,
                key=f"weight_cmap_chord_{field_key}",
            )
        with control_col3:
            outer_ring_radius = st.slider(
                "Source ring radius",
                min_value=0.85,
                max_value=4.00,
                value=4.00,
                step=0.01,
                key=f"outer_ring_radius_chord_{field_key}",
            )

        control_col4, control_col5 = st.columns([1, 1])
        with control_col4:
            show_dotted_ring = st.checkbox(
                "Show dotted source ring",
                value=True,
                key=f"show_dotted_ring_chord_{field_key}",
            )
        with control_col5:
            show_power_arc = st.checkbox(
                "Show outer power-group arc",
                value=True,
                key=f"show_power_arc_chord_{field_key}",
            )

        arc_col1, arc_col2 = st.columns([1, 1])
        with arc_col1:
            outer_power_arc_radius = st.slider(
                "Outer power-arc radius",
                min_value=float(outer_ring_radius + 0.10),
                max_value=5.00,
                value=4.72,
                step=0.01,
                key=f"outer_power_arc_radius_chord_{field_key}",
            )
        with arc_col2:
            power_label_radius = st.slider(
                "Outer power-label radius",
                min_value=float(outer_power_arc_radius + 0.05),
                max_value=6.00,
                value=5.33,
                step=0.01,
                key=f"outer_power_label_radius_chord_{field_key}",
            )

        chord_df = weight_df_sorted.head(top_n_chord).copy()
        chord_weights = chord_df["Final weight"].to_numpy(dtype=float)
        chord_max = float(np.nanmax(chord_weights)) if len(chord_weights) else 1.0
        if chord_max <= 0 or not np.isfinite(chord_max):
            chord_max = 1.0

        power_color_options = {
            "Purple": "#6A3D9A",
            "Crimson": "#B2182B",
            "Brown": "#6B4E16",
            "Charcoal": "#4D4D4D",
            "Deep Teal": "#006D77",
            "Magenta": "#C2185B",
            "Olive": "#6B8E23",
            "Navy": "#1F3A93",
            "Maroon": "#800000",
            "Turquoise": "#008B8B",
            "Gold": "#B8860B",
            "Slate": "#556B7A",
        }
        power_color_names = list(power_color_options.keys())
        default_power_color_names = {
            270: "Magenta",
            350: "Purple",
            370: "Magenta",
            400: "Brown",
            420: "Charcoal",
        }

        color_col1, color_col2 = st.columns(2)
        selected_power_color_names = {}
        for idx_power, power_value in enumerate([350, 370, 400, 420]):
            use_col = color_col1 if idx_power % 2 == 0 else color_col2
            with use_col:
                default_name = default_power_color_names.get(power_value, power_color_names[min(idx_power, len(power_color_names)-1)])
                default_index = power_color_names.index(default_name) if default_name in power_color_names else 0
                selected_power_color_names[power_value] = st.selectbox(
                    f"Power {power_value} W arc color",
                    power_color_names,
                    index=default_index,
                    key=f"power_arc_color_{power_value}_{field_key}",
                )

        power_palette = {
            power_value: power_color_options[selected_power_color_names[power_value]]
            for power_value in [350, 370, 400, 420]
        }
        power_fallback = list(power_color_options.values())
        speed_palette = {
            45: "#1F77B4",
            50: "#2CA02C",
            60: "#FF7F0E",
            70: "#D62788",
        }
        speed_fallback = ["#1F77B4", "#2CA02C", "#FF7F0E", "#D62788", "#17BECF", "#9467BD"]
        ctf_symbols = {
            0: "circle",
            1: "square",
            2: "diamond",
            3: "triangle-up",
            4: "hexagon",
        }
        ctf_fallback = ["circle", "square", "diamond", "triangle-up", "hexagon", "star"]

        def _power_color(power_value):
            p = int(power_value)
            if p in power_palette:
                return power_palette[p]
            unique_power_vals = sorted(chord_df["P (W)"].astype(int).unique().tolist())
            return power_fallback[unique_power_vals.index(p) % len(power_fallback)]

        def _speed_color(speed_value):
            s = int(speed_value)
            if s in speed_palette:
                return speed_palette[s]
            unique_speed_vals = sorted(chord_df["v (cm/s)"].astype(int).unique().tolist())
            return speed_fallback[unique_speed_vals.index(s) % len(speed_fallback)]

        def _ctf_symbol(tf_value):
            tfv = int(tf_value)
            if tfv in ctf_symbols:
                return ctf_symbols[tfv]
            unique_tfv = sorted(chord_df["cTF"].astype(int).unique().tolist())
            return ctf_fallback[unique_tfv.index(tfv) % len(ctf_fallback)]

        def _weight_hex(value):
            cmap = mpl.colormaps.get_cmap(weight_cmap)
            if chord_max <= 0:
                norm_val = 0.0
            else:
                norm_val = float(np.clip(value / chord_max, 0.0, 1.0))
            return mpl.colors.to_hex(cmap(norm_val), keep_alpha=False)

        def _rgba_from_any(color_value, alpha=1.0):
            rgba = mpl.colors.to_rgba(color_value, alpha=alpha)
            return f"rgba({int(round(rgba[0] * 255))}, {int(round(rgba[1] * 255))}, {int(round(rgba[2] * 255))}, {rgba[3]:.3f})"

        def _bezier_curve(x0, y0, bend=0.56, swirl=0.18, n=180):
            p0 = np.array([x0, y0], dtype=float)
            p3 = np.array([0.0, 0.0], dtype=float)
            r = float(np.hypot(x0, y0))
            if r < 1e-12:
                return np.zeros((n, 2), dtype=float)

            radial = p0 / r
            tangent = np.array([-radial[1], radial[0]], dtype=float)
            sign = 1.0 if (x0 >= 0 and y0 >= 0) or (x0 < 0 and y0 < 0) else -1.0
            swirl_mag = swirl * r

            p1 = p0 * (1.0 - 0.28 * bend) + sign * tangent * swirl_mag
            p2 = p0 * (0.22 + 0.10 * bend) - sign * tangent * (0.62 * swirl_mag)

            t = np.linspace(0.0, 1.0, n)[:, None]
            curve = (
                (1 - t) ** 3 * p0
                + 3 * (1 - t) ** 2 * t * p1
                + 3 * (1 - t) * t ** 2 * p2
                + t ** 3 * p3
            )
            return curve

        node_hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Power: %{customdata[1]:.0f} W<br>"
            "Scan speed: %{customdata[2]:.0f} cm/s<br>"
            "cTF: %{customdata[3]:.0f}<br>"
            "Co: %{customdata[4]:.3f}<br>"
            "Cr: %{customdata[5]:.3f}<br>"
            "Fe: %{customdata[6]:.3f}<br>"
            "Ni: %{customdata[7]:.3f}<br>"
            "Composition distance: %{customdata[8]:.5f}<br>"
            "Normalized P-v distance: %{customdata[9]:.5f}<br>"
            "Attention: %{customdata[10]:.5f}<br>"
            "P-v Gaussian: %{customdata[11]:.5f}<br>"
            "Composition: %{customdata[12]:.5f}<br>"
            "<b>Final weight: %{customdata[13]:.5f}</b>"
            "<extra></extra>"
        )

        def _node_customdata(df):
            return np.column_stack([
                df["Source"].to_numpy(),
                df["P (W)"].to_numpy(dtype=float),
                df["v (cm/s)"].to_numpy(dtype=float),
                df["cTF"].to_numpy(dtype=float),
                df["Co"].to_numpy(dtype=float),
                df["Cr"].to_numpy(dtype=float),
                df["Fe"].to_numpy(dtype=float),
                df["Ni"].to_numpy(dtype=float),
                df["Composition distance"].to_numpy(dtype=float),
                df["Normalized P-v distance"].to_numpy(dtype=float),
                df["Attention"].to_numpy(dtype=float),
                df["P-v Gaussian"].to_numpy(dtype=float),
                df["Composition"].to_numpy(dtype=float),
                df["Final weight"].to_numpy(dtype=float),
            ])

        def _add_target_node(fig):
            fig.add_trace(go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers",
                marker=dict(
                    size=46,
                    color="rgba(220, 0, 0, 0.98)",
                    symbol="star",
                    line=dict(color="black", width=3.4),
                ),
                hovertemplate=(
                    "<b>Target prediction</b><br>"
                    f"Power: {float(p_target):.2f} W<br>"
                    f"Scan speed: {float(v_target_cm_s):.2f} cm/s<br>"
                    f"Co: {tgt_comp[0]:.3f}<br>"
                    f"Cr: {tgt_comp[1]:.3f}<br>"
                    f"Fe: {tgt_comp[2]:.3f}<br>"
                    f"Ni: {tgt_comp[3]:.3f}"
                    "<extra></extra>"
                ),
                name="Target",
                showlegend=False,
            ))

        def _add_dotted_ring(fig, radius=1.0):
            theta = np.linspace(0, 2 * np.pi, 720)
            fig.add_trace(go.Scatter(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                mode="lines",
                line=dict(color="rgba(0,0,0,0.58)", width=4.0, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            ))

        def _add_power_arcs(fig, sector_bounds, radius):
            for power_value, (a0, a1) in sector_bounds.items():
                theta = np.linspace(a0, a1, 220)
                fig.add_trace(go.Scatter(
                    x=radius * np.cos(theta),
                    y=radius * np.sin(theta),
                    mode="lines",
                    line=dict(color=_power_color(power_value), width=18),
                    hovertemplate=f"<b>Power group</b><br>{int(power_value)} W<extra></extra>",
                    showlegend=False,
                ))

        def _add_speed_fill_legend(fig, speed_values):
            for speed_value in sorted(set(int(v) for v in speed_values)):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(
                        size=18, color=_speed_color(speed_value),
                        symbol="circle", line=dict(color="black", width=1.6),
                    ),
                    name=f"{int(speed_value)} cm/s",
                    legendgroup="speed_legend",
                    legendgrouptitle_text="Scan speed",
                    showlegend=True,
                    hoverinfo="skip",
                ))

        def _add_ctf_shape_legend(fig, ctf_values):
            for tf_value in sorted(set(int(v) for v in ctf_values)):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(
                        size=18, color="rgba(200,200,200,0.8)", symbol=_ctf_symbol(tf_value),
                        line=dict(color="black", width=1.8),
                    ),
                    name=f"cTF {int(tf_value)}",
                    legendgroup="ctf_legend",
                    legendgrouptitle_text="cTF",
                    showlegend=True,
                    hoverinfo="skip",
                ))

        def _add_weight_colorbar(fig):
            fig.add_trace(go.Scatter(
                x=[10], y=[10], mode="markers",
                marker=dict(
                    size=0.1, opacity=0.0, color=[0.0], cmin=0.0, cmax=chord_max,
                    colorscale=mpl_to_plotly_colorscale(weight_cmap),
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="<b>Final<br>weight</b>", font=dict(size=21, color="black", family="Arial Black")),
                        tickfont=dict(size=17, color="black", family="Arial Black"),
                        thickness=78,
                        len=0.44,
                        y=0.56,
                        outlinewidth=2.8,
                        outlinecolor="black",
                        bgcolor="rgba(255,255,255,0.94)",
                        x=1.03,
                    ),
                ),
                hoverinfo="skip", showlegend=False,
            ))

        def _format_chord_layout(fig, height=1170, radial_extent=2.2):
            fig.update_layout(
                title=None,
                height=height,
                margin=dict(l=20, r=120, t=35, b=40),
                xaxis=dict(visible=False, range=[-radial_extent, radial_extent], scaleanchor="y", scaleratio=1),
                yaxis=dict(visible=False, range=[-radial_extent, radial_extent]),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(size=18, color="black", family="Arial Black"),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.04,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=15, color="black", family="Arial Black"),
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="black",
                    borderwidth=1.8,
                ),
            )

        fig_chord = go.Figure()
        df_sector = chord_df.sort_values(["P (W)", "v (cm/s)", "cTF"]).reset_index(drop=True)
        node_x, node_y = [], []
        sector_powers = sorted(df_sector["P (W)"].astype(int).unique().tolist())
        gap = np.deg2rad(16)
        sector_width = (2 * np.pi - gap * len(sector_powers)) / max(len(sector_powers), 1)
        start_angle = np.pi / 2
        sector_bounds = {}
        sector_mid_angles = {}

        for s_idx, power_value in enumerate(sector_powers):
            group_idx = df_sector.index[df_sector["P (W)"].astype(int) == power_value].tolist()
            a0 = start_angle - s_idx * (sector_width + gap)
            a1 = a0 - sector_width
            sector_bounds[int(power_value)] = (a0, a1)
            sector_mid_angles[int(power_value)] = 0.5 * (a0 + a1)
            if len(group_idx) == 1:
                group_angles = [0.5 * (a0 + a1)]
            else:
                group_angles = np.linspace(a0 - 0.10, a1 + 0.10, len(group_idx)).tolist()
            for idx, ang in zip(group_idx, group_angles):
                node_x.append((idx, outer_ring_radius * np.cos(ang)))
                node_y.append((idx, outer_ring_radius * np.sin(ang)))


        node_x = np.array([v for _, v in sorted(node_x)], dtype=float)
        node_y = np.array([v for _, v in sorted(node_y)], dtype=float)

        if show_power_arc:
            _add_power_arcs(fig_chord, sector_bounds, radius=outer_power_arc_radius)
            for power_value in sector_powers:
                mid = sector_mid_angles[int(power_value)]
                fig_chord.add_annotation(
                    x=power_label_radius * np.cos(mid),
                    y=power_label_radius * np.sin(mid),
                    text=f"<b>{int(power_value)} W</b>",
                    showarrow=False,
                    font=dict(size=20, color="black", family="Arial Black"),
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="black",
                    borderwidth=2.0,
                )

        if show_dotted_ring:
            _add_dotted_ring(fig_chord, radius=outer_ring_radius)

        weights = df_sector["Final weight"].to_numpy(dtype=float)
        wmax = float(np.nanmax(weights)) if len(weights) else 1.0
        if wmax <= 0 or not np.isfinite(wmax):
            wmax = 1.0

        for i, (_, row) in enumerate(df_sector.iterrows()):
            curve = _bezier_curve(node_x[i], node_y[i], bend=0.56, swirl=0.22)
            rel = float(row["Final weight"] / wmax)
            width = 3.2 + 22.0 * rel
            opacity = 0.24 + 0.72 * rel
            link_color = _rgba_from_any(_weight_hex(row["Final weight"]), opacity)
            fig_chord.add_trace(go.Scatter(
                x=curve[:, 0],
                y=curve[:, 1],
                mode="lines",
                line=dict(width=width, color=link_color),
                hovertemplate=(
                    f"<b>{row['Source']} → Target</b><br>"
                    f"Power: {int(row['P (W)'])} W<br>"
                    f"Scan speed: {int(row['v (cm/s)'])} cm/s<br>"
                    f"cTF: {int(row['cTF'])}<br>"
                    f"Final weight: {row['Final weight']:.5f}<br>"
                    f"Attention: {row['Attention']:.5f}<br>"
                    f"P-v Gaussian: {row['P-v Gaussian']:.5f}<br>"
                    f"Composition: {row['Composition']:.5f}<br>"
                    f"Composition distance: {row['Composition distance']:.5f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        node_symbols = [_ctf_symbol(v) for v in df_sector["cTF"].to_numpy()]
        node_fill_colors = [_speed_color(v) for v in df_sector["v (cm/s)"].to_numpy()]
        node_sizes = 30 + 54 * (weights / wmax)

        fig_chord.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=node_fill_colors,
                symbol=node_symbols,
                opacity=0.99,
                line=dict(color="black", width=3.0),
            ),
            customdata=_node_customdata(df_sector),
            hovertemplate=node_hovertemplate,
            name="Source simulations",
            showlegend=False,
        ))

        _add_weight_colorbar(fig_chord)
        _add_target_node(fig_chord)
        _add_speed_fill_legend(fig_chord, df_sector["v (cm/s)"].astype(int).unique().tolist())
        _add_ctf_shape_legend(fig_chord, df_sector["cTF"].astype(int).unique().tolist())
        chord_extent = max(2.2, (power_label_radius + 0.35) if show_power_arc else (outer_ring_radius + 0.45))
        _format_chord_layout(fig_chord, radial_extent=chord_extent)

        render_plotly_with_live_transparent_download(
            fig_chord,
            filename=f"{target_download_prefix()}_{field_key.lower()}_chord_contribution_map.png",
            button_label="Download transparent PNG",
            key=f"chord_power_sector_final_{field_key}",
            height=1210,
            scale=3,
        )

    # ------------------------------------------------------------
    # 3. Weight decomposition plot
    # ------------------------------------------------------------
    with viz_tab3:
        top_n = st.slider(
            "Number of strongest source simulations to show",
            min_value=1,
            max_value=len(weight_df_sorted),
            value=min(8, len(weight_df_sorted)),
            step=1,
            key=f"top_n_weight_viz_{field_key}",
        )
        top_df = weight_df_sorted.head(top_n).copy()

        fig_bar = go.Figure()
        bar_colors = {
            "Attention": "#1f77b4",
            "P-v Gaussian": "#ff7f0e",
            "Composition": "#2ca02c",
            "Final weight": "#d62728",
        }
        for col in ["Attention", "P-v Gaussian", "Composition", "Final weight"]:
            fig_bar.add_trace(go.Bar(
                x=top_df["Source"],
                y=top_df[col],
                name=col,
                marker=dict(color=bar_colors[col], line=dict(color="black", width=1.6)),
            ))

        fig_bar.update_layout(
            title=dict(
                text=(
                    "<b>Weight decomposition for strongest source simulations</b>"
                    "<br><sup><b>Final weight = attention × process Gaussian × composition similarity, then normalized</b></sup>"
                ),
                font=dict(size=28, color="black", family="Arial Black"),
                x=0.5,
            ),
            xaxis_title="<b>Source simulation</b>",
            yaxis_title="<b>Weight value</b>",
            barmode="group",
            height=690,
            margin=dict(l=25, r=25, t=110, b=120),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=17, color="black", family="Arial Black"),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="black",
                borderwidth=1.6,
            ),
            font=dict(size=18, color="black", family="Arial Black"),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        fig_bar.update_xaxes(
            tickangle=-30,
            tickfont=dict(size=16, color="black", family="Arial Black"),
            title=dict(font=dict(size=22, color="black", family="Arial Black")),
            linecolor="black",
            linewidth=3,
            mirror=True,
            gridcolor="rgba(20,20,20,0.22)",
            gridwidth=1.8,
        )
        fig_bar.update_yaxes(
            tickfont=dict(size=16, color="black", family="Arial Black"),
            title=dict(font=dict(size=22, color="black", family="Arial Black")),
            linecolor="black",
            linewidth=3,
            mirror=True,
            gridcolor="rgba(20,20,20,0.28)",
            gridwidth=1.8,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2.0,
        )
        render_plotly_with_live_transparent_download(
            fig_bar,
            filename=f"{target_download_prefix()}_{field_key.lower()}_weight_decomposition.png",
            button_label="Download transparent PNG",
            key=f"weight_decomposition_{field_key}",
            height=760,
            scale=3,
        )

        csv_cols = [
            "Source", "P (W)", "v (cm/s)", "cTF",
            "Attention", "P-v Gaussian", "Composition", "Final weight",
        ]
        csv_buf = io.StringIO()
        top_df[csv_cols].to_csv(csv_buf, index=False)
        st.download_button(
            "Download bar-graph data as CSV",
            data=csv_buf.getvalue(),
            file_name=f"{target_download_prefix()}_{field_key.lower()}_weight_decomposition_data.csv",
            mime="text/csv",
            key=f"download_weight_decomp_csv_{field_key}",
        )

        st.markdown("#### Numerical source contribution table")
        st.dataframe(
            top_df[
                [
                    "Source", "P (W)", "v (cm/s)", "cTF", "Co", "Cr", "Fe", "Ni",
                    "Composition distance", "Normalized P-v distance",
                    "Attention", "P-v Gaussian", "Composition", "Final weight",
                ]
            ].style.format({
                "P (W)": "{:.0f}",
                "v (cm/s)": "{:.0f}",
                "Co": "{:.3f}",
                "Cr": "{:.3f}",
                "Fe": "{:.3f}",
                "Ni": "{:.3f}",
                "Composition distance": "{:.5f}",
                "Normalized P-v distance": "{:.5f}",
                "Attention": "{:.5f}",
                "P-v Gaussian": "{:.5f}",
                "Composition": "{:.5f}",
                "Final weight": "{:.5f}",
            }).bar(subset=["Final weight"], color="#5fba7d"),
            width='stretch',
        )


    # ------------------------------------------------------------
    # 4. Query-Key heatmap plot
    # ------------------------------------------------------------
    with viz_tab4:
        st.markdown("### Query and key heatmaps")
        st.caption(
            "These heatmaps show the learned linear projection weights that map the input process variables "
            "Power and Scan speed into the attention query and key feature space."
        )

        qk_ctrl1, qk_ctrl2 = st.columns([1.0, 0.85])
        with qk_ctrl1:
            qk_cmap = st.selectbox(
                "Heatmap colormap",
                CMAP_OPTIONS,
                index=CMAP_OPTIONS.index("balance") if "balance" in CMAP_OPTIONS else (CMAP_OPTIONS.index("RdBu") if "RdBu" in CMAP_OPTIONS else 0),
                key=f"qk_heatmap_cmap_tab_{field_key}",
            )
        with qk_ctrl2:
            qk_height = st.slider(
                "Figure height",
                min_value=500,
                max_value=950,
                value=700,
                step=25,
                key=f"qk_heatmap_height_tab_{field_key}",
            )

        q_mat = np.array(results["W_q_proc"], dtype=float)
        k_mat = np.array(results["W_k_proc"], dtype=float)
        global_abs = float(max(np.max(np.abs(q_mat)), np.max(np.abs(k_mat)), 1e-12))
        q_rows = [f"F{i+1}" for i in range(q_mat.shape[0])]
        k_rows = [f"F{i+1}" for i in range(k_mat.shape[0])]
        feat_cols = ["Power", "Scan speed"] if q_mat.shape[1] == 2 else [f"In {j+1}" for j in range(q_mat.shape[1])]

        fig_qk = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.13,
            subplot_titles=(
                "<b>W<sub>q</sub> — Query projection</b>",
                "<b>W<sub>k</sub> — Key projection</b>",
            ),
        )

        fig_qk.add_trace(
            go.Heatmap(
                z=q_mat,
                x=feat_cols,
                y=q_rows,
                zmin=-global_abs,
                zmax=global_abs,
                colorscale=mpl_to_plotly_colorscale(qk_cmap),
                colorbar=dict(
                    title=dict(text="<b>Query<br>weight</b>", font=dict(size=20, color="black", family="Arial Black")),
                    thickness=42,
                    len=0.72,
                    outlinewidth=2.4,
                    outlinecolor="black",
                    tickfont=dict(size=16, color="black", family="Arial Black"),
                    x=0.46,
                    y=0.50,
                ),
                hovertemplate=(
                    "<b>Query projection</b><br>"
                    "Projected feature: %{y}<br>"
                    "Input variable: %{x}<br>"
                    "Value: %{z:.6f}<extra></extra>"
                ),
                showscale=True,
            ),
            row=1, col=1,
        )

        fig_qk.add_trace(
            go.Heatmap(
                z=k_mat,
                x=feat_cols,
                y=k_rows,
                zmin=-global_abs,
                zmax=global_abs,
                colorscale=mpl_to_plotly_colorscale(qk_cmap),
                colorbar=dict(
                    title=dict(text="<b>Key<br>weight</b>", font=dict(size=20, color="black", family="Arial Black")),
                    thickness=42,
                    len=0.72,
                    outlinewidth=2.4,
                    outlinecolor="black",
                    tickfont=dict(size=16, color="black", family="Arial Black"),
                    x=1.02,
                    y=0.50,
                ),
                hovertemplate=(
                    "<b>Key projection</b><br>"
                    "Projected feature: %{y}<br>"
                    "Input variable: %{x}<br>"
                    "Value: %{z:.6f}<extra></extra>"
                ),
                showscale=True,
            ),
            row=1, col=2,
        )
        fig_qk.update_layout(
            title=dict(
                text=(
                    "<b>Query and key projection matrices</b>"
                    "<br><sup><b>Learned mapping from process variables to attention feature space</b></sup>"
                ),
                x=0.5,
                font=dict(size=30, color="black", family="Arial Black"),
            ),
            height=qk_height,
            margin=dict(l=25, r=65, t=115, b=55),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=18, color="black", family="Arial Black"),
        )
        fig_qk.update_annotations(font=dict(size=20, color="black", family="Arial Black"))
        fig_qk.update_xaxes(
            showgrid=True,
            gridcolor="rgba(20,20,20,0.28)",
            gridwidth=1.8,
            linecolor="black",
            linewidth=3.0,
            mirror=True,
            tickfont=dict(size=18, color="black", family="Arial Black"),
            title=dict(font=dict(size=20, color="black", family="Arial Black")),
        )
        fig_qk.update_yaxes(
            showgrid=True,
            gridcolor="rgba(20,20,20,0.28)",
            gridwidth=1.8,
            linecolor="black",
            linewidth=3.0,
            mirror=True,
            tickfont=dict(size=18, color="black", family="Arial Black"),
            title=dict(font=dict(size=20, color="black", family="Arial Black")),
            autorange="reversed",
        )

        render_plotly_with_live_transparent_download(
            fig_qk,
            filename=f"{target_download_prefix()}_{field_key.lower()}_query_key_heatmaps.png",
            button_label="Download transparent PNG",
            key=f"qk_heatmaps_tab_{field_key}",
            height=int(qk_height + 70),
            scale=3,
        )

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

    plot_source_target_weight_visualization(cached_sources, results, field_key)

    with st.expander("Parameter-space weight plot and query/key heatmaps", expanded=False):
        col_param, col_qk = st.columns([1.20, 1.35])

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
            st.markdown("#### Normalized parameter-space map")
            fig_param = go.Figure()
            fig_param.add_trace(go.Scatter3d(
                x=v_norm,
                y=p_norm,
                z=tf_norm,
                mode="markers",
                marker=dict(
                    size=8,
                    color=results["combined_weights"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="<b>Final<br>weight</b>", font=dict(size=16, color="black")),
                        thickness=22,
                        len=0.68,
                        tickfont=dict(size=13, color="black"),
                        outlinewidth=1.6,
                        outlinecolor="black",
                    ),
                    opacity=0.95,
                    line=dict(color="black", width=1.2),
                ),
                hovertemplate=(
                    "Power=%{customdata[0]} W<br>"
                    "Scan speed=%{customdata[1]} cm/s<br>"
                    "cTF=%{customdata[2]}<br>"
                    "Weight=%{marker.color:.5f}<extra></extra>"
                ),
                customdata=np.column_stack([src_p, src_v, src_tf]),
                name="Sources",
                showlegend=False,
            ))
            fig_param.add_trace(go.Scatter3d(
                x=[(float(v_target_cm_s) - v_min) / (v_max - v_min + 1e-8)],
                y=[(float(p_target) - p_min) / (p_max - p_min + 1e-8)],
                z=[target_tf_norm],
                mode="markers",
                marker=dict(size=9, symbol="diamond", color="red", line=dict(color="black", width=1.2)),
                hovertemplate=f"Target<br>Power={float(p_target):.2f} W<br>Scan speed={float(v_target_cm_s):.2f} cm/s<br>Nearest cTF={nearest_tf}<extra></extra>",
                name="Target",
                showlegend=False,
            ))
            fig_param.update_layout(
                title=dict(text="<b>3D normalized parameter space</b>", x=0.5, font=dict(size=18, color="black", family="Arial Black")),
                scene=dict(
                    xaxis=dict(title=dict(text="Normalized scan speed", font=dict(size=14, color="black")), tickfont=dict(size=12, color="black"), linecolor="black", linewidth=3, showbackground=True, backgroundcolor="rgba(245,245,245,0.75)"),
                    yaxis=dict(title=dict(text="Normalized power", font=dict(size=14, color="black")), tickfont=dict(size=12, color="black"), linecolor="black", linewidth=3, showbackground=True, backgroundcolor="rgba(245,245,245,0.75)"),
                    zaxis=dict(title=dict(text="cTF index / 4", font=dict(size=14, color="black")), tickfont=dict(size=12, color="black"), linecolor="black", linewidth=3, showbackground=True, backgroundcolor="rgba(245,245,245,0.75)", tickmode="array", tickvals=[0, 0.25, 0.5, 0.75, 1.0], ticktext=["cTF0", "cTF1", "cTF2", "cTF3", "cTF4"]),
                ),
                margin=dict(l=0, r=0, t=50, b=0),
                height=560,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(size=13, color="black", family="Arial Black"),
            )
            st.plotly_chart(fig_param, width='stretch', key=f"param_plot_3d_{field_key}")

        with col_qk:
            st.markdown("#### Query and key heatmaps")
            qk_cmap = st.selectbox(
                "Heatmap colormap",
                CMAP_OPTIONS,
                index=CMAP_OPTIONS.index("balance") if "balance" in CMAP_OPTIONS else (CMAP_OPTIONS.index("RdBu") if "RdBu" in CMAP_OPTIONS else 0),
                key=f"qk_heatmap_cmap_{field_key}",
            )

            q_mat = np.array(results["W_q_proc"], dtype=float)
            k_mat = np.array(results["W_k_proc"], dtype=float)
            global_abs = float(max(np.max(np.abs(q_mat)), np.max(np.abs(k_mat)), 1e-12))
            q_rows = [f"F{i+1}" for i in range(q_mat.shape[0])]
            k_rows = [f"F{i+1}" for i in range(k_mat.shape[0])]
            feat_cols = ["Power", "Scan speed"] if q_mat.shape[1] == 2 else [f"In {j+1}" for j in range(q_mat.shape[1])]

            fig_qk = make_subplots(
                rows=1,
                cols=2,
                horizontal_spacing=0.15,
                subplot_titles=("<b>W<sub>q</sub> (Query projection)</b>", "<b>W<sub>k</sub> (Key projection)</b>"),
            )

            fig_qk.add_trace(
                go.Heatmap(
                    z=q_mat,
                    x=feat_cols,
                    y=q_rows,
                    zmin=-global_abs,
                    zmax=global_abs,
                    colorscale=mpl_to_plotly_colorscale(qk_cmap),
                    colorbar=dict(
                        title=dict(text="<b>Weight</b>", font=dict(size=16, color="black", family="Arial Black")),
                        thickness=26,
                        len=0.76,
                        outlinewidth=1.8,
                        outlinecolor="black",
                        tickfont=dict(size=13, color="black", family="Arial Black"),
                        x=0.46,
                        y=0.50,
                    ),
                    hovertemplate="<b>Query</b><br>Projected feature: %{y}<br>Input: %{x}<br>Value: %{z:.5f}<extra></extra>",
                    showscale=True,
                ),
                row=1, col=1,
            )
            fig_qk.add_trace(
                go.Heatmap(
                    z=k_mat,
                    x=feat_cols,
                    y=k_rows,
                    zmin=-global_abs,
                    zmax=global_abs,
                    colorscale=mpl_to_plotly_colorscale(qk_cmap),
                    colorbar=dict(
                        title=dict(text="<b>Weight</b>", font=dict(size=16, color="black", family="Arial Black")),
                        thickness=26,
                        len=0.76,
                        outlinewidth=1.8,
                        outlinecolor="black",
                        tickfont=dict(size=13, color="black", family="Arial Black"),
                        x=1.02,
                        y=0.50,
                    ),
                    hovertemplate="<b>Key</b><br>Projected feature: %{y}<br>Input: %{x}<br>Value: %{z:.5f}<extra></extra>",
                    showscale=True,
                ),
                row=1, col=2,
            )
            fig_qk.update_layout(
                title=dict(
                    text="<b>Query and key projection matrices</b><br><sup><b>These heatmaps show how Power and Scan speed are projected into the attention space</b></sup>",
                    x=0.5,
                    font=dict(size=22, color="black", family="Arial Black"),
                ),
                height=610,
                margin=dict(l=20, r=40, t=95, b=40),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(size=13, color="black", family="Arial Black"),
            )
            fig_qk.update_xaxes(showgrid=True, gridcolor="rgba(20,20,20,0.18)", gridwidth=1.2, linecolor="black", linewidth=2.0, tickfont=dict(size=12, color="black", family="Arial Black"), title=dict(font=dict(size=14, color="black", family="Arial Black")))
            fig_qk.update_yaxes(showgrid=True, gridcolor="rgba(20,20,20,0.18)", gridwidth=1.2, linecolor="black", linewidth=2.0, tickfont=dict(size=12, color="black", family="Arial Black"), title=dict(font=dict(size=14, color="black", family="Arial Black")), autorange="reversed")

            render_plotly_with_live_transparent_download(
                fig_qk,
                filename=f"{target_download_prefix()}_{field_key.lower()}_query_key_heatmaps.png",
                button_label="Download transparent PNG",
                key=f"qk_heatmaps_{field_key}",
                height=660,
                scale=3,
            )

    st.markdown(f"### {label} animation")

    if is_phase:
        anim_field = np.array([(predicted[k] >= phase_threshold).astype(float) for k in range(Nt)])
        plot_unit = "phase"
        stat_field = predicted
        st.write(
            f"Continuous LIQUID fraction: min={stat_field.min():.4f}, "
            f"max={stat_field.max():.4f}, mean={stat_field.mean():.4f}. "
            f"Binary animation uses threshold={phase_threshold:.2f}."
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
        static_slider_kwargs = {"key": static_key}
        if static_key not in st.session_state:
            static_slider_kwargs["value"] = Nt // 2
        t_idx_static = int(st.slider("Time index", 0, Nt - 1, **static_slider_kwargs))
        current = predicted[t_idx_static]

        if is_phase:
            display_field = (current >= phase_threshold).astype(float)
            stat_field = predicted[t_idx_static]
            st.write(
                f"Continuous LIQUID fraction @ t={t_idx_static}: min={stat_field.min():.4f}, "
                f"max={stat_field.max():.4f}, mean={stat_field.mean():.4f}. "
                f"Binary static plot uses threshold={phase_threshold:.2f}."
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
    base_name = f"{target_download_prefix()}_{field_key.lower()}_pred"
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
            "composition": target_composition,
            "weights": results["combined_weights"],
        }
        if is_phase:
            extra["binary_phase"] = (predicted >= phase_threshold).astype(np.uint8)
            extra["threshold"] = phase_threshold
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
        "This tab predicts TEMP, ETALIQ and VEL together, then plots only the velocity field. "
        "Velocity is kept only inside the predicted LIQUID region; the FCC/solid region is set to 0. "
        "The LIQUID/FCC boundary and the 800 K / 1800 K temperature contours are overlaid on the velocity plot."
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
    eta_pred = np.clip(combo["ETALIQ"]["prediction"], 0.0, 1.0)
    vel_pred = combo["VEL"]["prediction"]

    if temp_pred.shape != eta_pred.shape or temp_pred.shape != vel_pred.shape:
        st.error(f"Predicted shapes must match. TEMP={temp_pred.shape}, ETALIQ={eta_pred.shape}, VEL={vel_pred.shape}")
        return

    Nt, Ny, Nx = vel_pred.shape

    st.markdown("### COMBO animation")
    combo_anim_fig = plot_combo_animation(temp_pred, eta_pred, vel_pred, vel_cmap, phase_threshold)
    st.plotly_chart(combo_anim_fig, width='stretch', key="combo_animation_plot")

    st.markdown("### Static COMBO plot and PNG export")
    combo_static_key = "combo_time_index"
    combo_slider_kwargs = {"key": combo_static_key}
    if combo_static_key not in st.session_state:
        combo_slider_kwargs["value"] = Nt // 2
    t_idx = int(st.slider("Time index", 0, Nt - 1, **combo_slider_kwargs))

    fig_combo, vel_masked, stats = plot_combo_velocity(temp_pred, eta_pred, vel_pred, t_idx, vel_cmap, phase_threshold)
    st.pyplot(fig_combo)

    st.write(
        f"At t={t_idx}: velocity range shown = {stats['vel_min']:.4g} to {stats['vel_max']:.4g}; "
        f"mean velocity in LIQUID = {stats['vel_mean_liquid']:.4g}; "
        f"LIQUID area fraction = {stats['liquid_fraction']:.4f}."
    )

    png_buffer = transparent_png_bytes(fig_combo)
    base_name = f"{target_download_prefix()}_combo_velocity_masked_t{t_idx}.png"
    st.download_button(
        "Download transparent PNG for this time index",
        data=png_buffer,
        file_name=base_name,
        mime="image/png",
        key="download_combo_png",
    )
    plt.close(fig_combo)

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
5. A Gaussian composition-similarity term compares the target `[Co, Cr, Fe, Ni]` against the source `cTF` liquid composition vector.
6. The final normalized hybrid weight is used to interpolate the full spatiotemporal field.

The plots are kept as direct static/animated field visualizations with manual colormap selection, temperature contours at 800 K and 1800 K, and `.npy/.npz` downloads.
"""
    )
