#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import math
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from streamlit_plotly_events import plotly_events

from scipy.interpolate import LinearNDInterpolator
from scipy.linalg import lstsq
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="CoCrFeNi Gibbs Energy Explorer",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Robust spherical harmonics import
# ============================================================
try:
    from scipy.special import sph_harm
    SCIPY_SH_AVAILABLE = True
except Exception:
    SCIPY_SH_AVAILABLE = False
    st.warning("scipy.special.sph_harm not available. Using NumPy fallback.")

    def legendre_p(l, m, x):
        x = np.clip(x, -1, 1)

        if m > l:
            return np.zeros_like(x)

        pmm = np.ones_like(x)

        if m > 0:
            somx2 = np.sqrt((1 - x) * (1 + x))
            fact = 1.0

            for _ in range(1, m + 1):
                pmm *= -fact * somx2
                fact += 2.0

        if l == m:
            return pmm

        pmmp1 = x * (2 * m + 1) * pmm

        if l == m + 1:
            return pmmp1

        pll = np.zeros((l - m + 1, len(x)))
        pll[0] = pmm
        pll[1] = pmmp1

        for ll in range(m + 2, l + 1):
            pll[ll - m] = (
                (2 * ll - 1) * x * pll[ll - m - 1]
                - (ll + m - 1) * pll[ll - m - 2]
            ) / (ll - m)

        return pll[l - m]

    def sph_harm(m, l, theta, phi):
        theta = np.asarray(theta)
        phi = np.asarray(phi)

        norm = np.sqrt(
            (2 * l + 1)
            / (4 * np.pi)
            * math.factorial(l - abs(m))
            / math.factorial(l + abs(m))
        )

        plm = legendre_p(l, abs(m), np.cos(phi))
        azimuth = np.exp(1j * m * theta)

        return norm * plm * azimuth * ((-1) ** m if m >= 0 else 1)


# ============================================================
# Constants
# ============================================================
COLORMAPS = sorted(
    list(
        set(
            [
                "Viridis",
                "Plasma",
                "Inferno",
                "Magma",
                "Cividis",
                "Turbo",
                "Blues",
                "BuGn",
                "BuPu",
                "GnBu",
                "Greens",
                "Greys",
                "Oranges",
                "OrRd",
                "PuBu",
                "PuBuGn",
                "PuRd",
                "Purples",
                "RdPu",
                "Reds",
                "YlGn",
                "YlGnBu",
                "YlOrBr",
                "YlOrRd",
                "BrBG",
                "PRGn",
                "PiYG",
                "PuOr",
                "RdBu",
                "RdGy",
                "RdYlBu",
                "RdYlGn",
                "Spectral",
                "Phase",
                "Twilight",
                "HSV",
                "Jet",
                "Rainbow",
                "Hot",
                "Cool",
                "Spring",
                "Summer",
                "Autumn",
                "Winter",
                "Bone",
                "Copper",
                "Cubehelix",
                "Terrain",
                "Ocean",
                "Sinebow",
                "Prism",
                "Flag",
                "Gnuplot",
                "Gnuplot2",
                "CMRmap",
                "Afmhot",
                "Gist_heat",
                "Gist_rainbow",
                "Gist_stern",
                "Gist_earth",
                "Gist_ncar",
                "Brg",
                "Bwr",
                "Seismic",
                "Coolwarm",
                "Blackbody",
                "Electric",
                "Algae",
                "Deep",
                "Dense",
                "Haline",
                "Ice",
                "Matter",
                "Speed",
                "Tempo",
                "Thermal",
                "Turbid",
                "Plotly3",
                "Portland",
                "Picnic",
                "Solar",
                "Balance",
                "Delta",
                "Curl",
                "IceFire",
                "Edge",
                "Fall",
                "Sunset",
                "Sunsetdark",
                "Teal",
                "Tealgrn",
                "Tropic",
                "Peach",
                "Oxy",
                "Mint",
                "Emrld",
                "Aggrnyl",
                "Agsunset",
                "Armyrose",
                "Bluered",
                "Blugrn",
                "Bluyl",
                "Brwnyl",
                "Burg",
                "Burgyl",
                "Darkmint",
                "Geysr",
                "Magenta",
                "Mrybm",
                "Mygbm",
                "Oryel",
                "Pinkyl",
                "Purp",
                "Purpor",
                "Redor",
                "Ylorrd",
                "Ylorbr",
                "Ylgnbu",
                "Ylgn",
            ]
        )
    )
)

SYMBOLS = [
    "circle",
    "diamond",
    "cross",
    "x",
    "star",
    "square",
    "pentagon",
    "hexagon",
    "hexagon2",
    "octagon",
    "star-diamond",
    "star-triangle-up",
    "star-square",
]


# ============================================================
# Coordinate transformations
# ============================================================
def cartesian_to_spherical(c1, c2, c3):
    r = np.sqrt(c1**2 + c2**2 + c3**2)
    safe_r = np.where(r == 0, 1e-12, r)
    theta = np.arctan2(c2, c1)
    phi = np.arccos(np.clip(c3 / safe_r, -1.0, 1.0))
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


# ============================================================
# Spherical harmonics utilities
# ============================================================
def compute_real_spherical_harmonics(l_max, theta, phi):
    n_pts = len(theta) if hasattr(theta, "__len__") else 1
    n_coeffs = (l_max + 1) ** 2
    Y = np.zeros((n_pts, n_coeffs))

    idx = 0

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                Y[:, idx] = np.sqrt(2) * sph_harm(abs(m), l, theta, phi).imag
            elif m > 0:
                Y[:, idx] = np.sqrt(2) * sph_harm(m, l, theta, phi).real
            else:
                Y[:, idx] = sph_harm(0, l, theta, phi).real

            idx += 1

    return Y


def fit_spherical_harmonics(G_values, theta, phi, l_max=4, reg_lambda=1e-6):
    Y = compute_real_spherical_harmonics(l_max, theta, phi)

    YtY = Y.T @ Y
    YtG = Y.T @ G_values
    n_coeffs = YtY.shape[0]

    coeffs, residuals, rank, s = lstsq(
        YtY + reg_lambda * np.eye(n_coeffs),
        YtG,
    )

    G_reconstructed = Y @ coeffs

    rmse = np.sqrt(np.mean((G_values - G_reconstructed) ** 2))
    condition_number = np.linalg.cond(YtY + reg_lambda * np.eye(n_coeffs))

    return coeffs, G_reconstructed, rmse, condition_number


def generate_sh_surface(coeffs, l_max, r_base=0.5, r_scale=0.3, resolution=60):
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0.01, np.pi - 0.01, resolution)

    Theta, Phi = np.meshgrid(theta, phi)

    G_sh = np.zeros_like(Theta)

    idx = 0

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                basis = np.sqrt(2) * sph_harm(abs(m), l, Theta, Phi).imag
            elif m > 0:
                basis = np.sqrt(2) * sph_harm(m, l, Theta, Phi).real
            else:
                basis = sph_harm(0, l, Theta, Phi).real

            G_sh += coeffs[idx] * basis
            idx += 1

    G_norm = (G_sh - G_sh.min()) / (G_sh.max() - G_sh.min() + 1e-10)
    r = r_base + r_scale * G_norm

    X = r * np.sin(Phi) * np.cos(Theta)
    Y = r * np.sin(Phi) * np.sin(Theta)
    Z = r * np.cos(Phi)

    return X, Y, Z, G_sh, Theta, Phi


def compute_sh_gradient(coeffs, l_max, theta, phi):
    eps = 1e-6

    def eval_sh(th, ph):
        values = np.zeros_like(th)
        idx = 0

        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                if m < 0:
                    basis = np.sqrt(2) * sph_harm(abs(m), l, th, ph).imag
                elif m > 0:
                    basis = np.sqrt(2) * sph_harm(m, l, th, ph).real
                else:
                    basis = sph_harm(0, l, th, ph).real

                values += coeffs[idx] * basis
                idx += 1

        return values

    G_center = eval_sh(theta, phi)
    G_theta_plus = eval_sh(theta + eps, phi)
    G_phi_plus = eval_sh(theta, phi + eps)

    dG_dtheta = (G_theta_plus - G_center) / eps
    dG_dphi = (G_phi_plus - G_center) / eps

    return dG_dtheta, dG_dphi


# ============================================================
# Data loading
# ============================================================
@st.cache_data(ttl=3600)
def load_gibbs_data(csv_folder, T_min_input, T_max_input, T_step_input):
    temperatures = list(range(T_min_input, T_max_input + 1, T_step_input))

    df_list = []
    missing_files = []

    for T in temperatures:
        file_path = os.path.join(csv_folder, f"Gibbs_{T}K.csv")

        if not os.path.exists(file_path):
            missing_files.append(file_path)
            continue

        try:
            df_T = pd.read_csv(file_path)

            required_cols = ["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"]
            missing_cols = [c for c in required_cols if c not in df_T.columns]

            if missing_cols:
                st.warning(f"Skipping {file_path}. Missing columns: {missing_cols}")
                continue

            df_T = df_T[required_cols].copy()
            df_T["T"] = T

            df_list.append(df_T)

        except Exception as e:
            st.warning(f"Skipping {file_path}: {e}")

    if len(df_list) == 0:
        return None, missing_files

    df_plot = pd.concat(df_list, ignore_index=True)

    df_plot["DeltaG"] = df_plot["G_FCC"] - df_plot["G_LIQ"]
    df_plot["G_stable"] = df_plot[["G_LIQ", "G_FCC"]].min(axis=1)
    df_plot["Stable_Phase"] = np.where(
        df_plot["G_LIQ"] <= df_plot["G_FCC"],
        "LIQUID",
        "FCC",
    )

    return df_plot, missing_files


@st.cache_data(ttl=1800)
def build_interpolators_for_T(df, T):
    df_T = df[df["T"] == T].copy()

    if len(df_T) == 0:
        return None, None

    pts = df_T[["Co", "Cr", "Fe"]].values

    interp_liq = LinearNDInterpolator(
        pts,
        df_T["G_LIQ"].values,
        fill_value=np.nan,
    )

    interp_fcc = LinearNDInterpolator(
        pts,
        df_T["G_FCC"].values,
        fill_value=np.nan,
    )

    return interp_liq, interp_fcc


@st.cache_data(ttl=1800)
def build_interpolated_grid(df, T_val, grid_res):
    interp_liq, interp_fcc = build_interpolators_for_T(df, T_val)

    if interp_liq is None or interp_fcc is None:
        return None

    x = np.linspace(0.0, 1.0, grid_res)

    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")

    grid_pts = np.column_stack(
        [
            Xco.ravel(),
            Xcr.ravel(),
            Xfe.ravel(),
        ]
    )

    valid_mask = np.sum(grid_pts, axis=1) <= 1.0
    pts_valid = grid_pts[valid_mask]

    G_liq = interp_liq(pts_valid)
    G_fcc = interp_fcc(pts_valid)

    valid_eval = ~np.isnan(G_liq) & ~np.isnan(G_fcc)

    pts = pts_valid[valid_eval]
    G_liq = G_liq[valid_eval]
    G_fcc = G_fcc[valid_eval]

    if len(pts) == 0:
        return None

    G_stable = np.minimum(G_liq, G_fcc)
    stable_label = np.where(G_liq <= G_fcc, "LIQUID", "FCC")

    Ni = 1.0 - pts[:, 0] - pts[:, 1] - pts[:, 2]
    DeltaG = G_fcc - G_liq

    df_interp = pd.DataFrame(
        {
            "Co": pts[:, 0],
            "Cr": pts[:, 1],
            "Fe": pts[:, 2],
            "Ni": Ni,
            "T": T_val,
            "G_LIQ": G_liq,
            "G_FCC": G_fcc,
            "DeltaG": DeltaG,
            "G_stable": G_stable,
            "Stable_Phase": stable_label,
        }
    )

    return df_interp


@st.cache_data(ttl=3600)
def compute_gibbs_tensor(df, comp_grid_res=15):
    T_array = np.array(sorted(df["T"].unique()))

    comp_vals = np.linspace(0, 1, comp_grid_res)

    Co_grid, Cr_grid, Fe_grid = np.meshgrid(
        comp_vals,
        comp_vals,
        comp_vals,
        indexing="ij",
    )

    comp_pts = np.column_stack(
        [
            Co_grid.ravel(),
            Cr_grid.ravel(),
            Fe_grid.ravel(),
        ]
    )

    valid_mask = np.sum(comp_pts, axis=1) <= 1.0
    valid_grid_mask = valid_mask.reshape(Co_grid.shape)

    n_T = len(T_array)

    G_LIQ = np.full(
        (n_T, comp_grid_res, comp_grid_res, comp_grid_res),
        np.nan,
    )

    G_FCC = np.full(
        (n_T, comp_grid_res, comp_grid_res, comp_grid_res),
        np.nan,
    )

    phase_matrix = np.full(
        (n_T, comp_grid_res, comp_grid_res, comp_grid_res),
        -1,
        dtype=int,
    )

    for t_idx, T_val in enumerate(T_array):
        df_T = df[df["T"] == T_val].copy()

        if len(df_T) < 4:
            continue

        pts = df_T[["Co", "Cr", "Fe"]].values

        interp_liq = LinearNDInterpolator(
            pts,
            df_T["G_LIQ"].values,
            fill_value=np.nan,
        )

        interp_fcc = LinearNDInterpolator(
            pts,
            df_T["G_FCC"].values,
            fill_value=np.nan,
        )

        g_liq_vals = interp_liq(comp_pts[valid_mask])
        g_fcc_vals = interp_fcc(comp_pts[valid_mask])

        G_LIQ[t_idx][valid_grid_mask] = g_liq_vals
        G_FCC[t_idx][valid_grid_mask] = g_fcc_vals

        valid_eval = ~np.isnan(g_liq_vals) & ~np.isnan(g_fcc_vals)

        phase_flat = np.full(np.sum(valid_mask), -1, dtype=int)
        phase_flat[valid_eval] = np.where(
            g_liq_vals[valid_eval] <= g_fcc_vals[valid_eval],
            0,
            1,
        )

        phase_matrix[t_idx][valid_grid_mask] = phase_flat

    G_stable = np.minimum(G_LIQ, G_FCC)

    return {
        "G_LIQ": G_LIQ,
        "G_FCC": G_FCC,
        "G_stable": G_stable,
        "phase_matrix": phase_matrix,
        "composition_grid": (Co_grid, Cr_grid, Fe_grid),
        "T_array": T_array,
        "comp_vals": comp_vals,
    }


def detect_phase_boundaries(phase_matrix, comp_grid, T_idx, smoothing_sigma=1.0):
    Co_grid, Cr_grid, Fe_grid = comp_grid

    phase_slice = phase_matrix[T_idx].astype(float)
    phase_slice[phase_slice < 0] = np.nan

    if np.all(np.isnan(phase_slice)):
        return np.empty((0, 3)), np.empty(0)

    phase_filled = np.nan_to_num(
        phase_slice,
        nan=np.nanmean(phase_slice),
    )

    phase_smooth = gaussian_filter(phase_filled, sigma=smoothing_sigma)

    grad_co = np.gradient(phase_smooth, axis=0)
    grad_cr = np.gradient(phase_smooth, axis=1)
    grad_fe = np.gradient(phase_smooth, axis=2)

    grad_mag = np.sqrt(grad_co**2 + grad_cr**2 + grad_fe**2)

    valid_mask = ~np.isnan(grad_mag)

    if not np.any(valid_mask):
        return np.empty((0, 3)), np.empty(0)

    threshold = np.percentile(grad_mag[valid_mask], 90)
    boundary_mask = (grad_mag > threshold) & valid_mask

    boundary_points = np.column_stack(
        [
            Co_grid[boundary_mask],
            Cr_grid[boundary_mask],
            Fe_grid[boundary_mask],
        ]
    )

    boundary_conf = grad_mag[boundary_mask]

    return boundary_points, boundary_conf


# ============================================================
# Main title
# ============================================================
st.title("🔷 Co-Cr-Fe-Ni Gibbs Free Energy Explorer")

st.markdown(
    r"""
This app contains two integrated visualizations:

**TAB 1:** Interactive 8D parallel plot of discrete thermodynamic states.  
**TAB 2:** Interpolated Gibbs-energy tensor visualization with coordinate transforms, phase stability, spherical harmonics, and tensor export.

Stable phase condition:

$$
G_{\mathrm{stable}} = \min(G_{\mathrm{LIQ}}, G_{\mathrm{FCC}})
$$
"""
)


# ============================================================
# Global sidebar
# ============================================================
with st.sidebar:
    st.header("📁 Data Settings")

    csv_folder = st.text_input("CSV folder", "csv_files")

    T_min_input = st.number_input("Minimum temperature [K]", value=300, step=100)
    T_max_input = st.number_input("Maximum temperature [K]", value=3300, step=100)
    T_step_input = st.number_input("Temperature step [K]", value=100, step=100)

    st.divider()

    st.header("📈 TAB 1 Parallel Plot")

    max_rows = st.slider(
        "Maximum plotted strands",
        min_value=50,
        max_value=3000,
        value=300,
        step=50,
    )

    random_seed = st.number_input("Random seed", value=42, step=1)

    line_width = st.slider("Line width", 0.2, 3.0, 0.8, 0.1)
    line_opacity = st.slider("Line opacity", 0.05, 1.0, 0.85, 0.05)

    show_raw_data = st.checkbox("Show raw dataframe", value=False)
    show_normalized_data = st.checkbox("Show normalized dataframe", value=False)


# ============================================================
# Load data
# ============================================================
df_plot, missing_files = load_gibbs_data(
    csv_folder,
    int(T_min_input),
    int(T_max_input),
    int(T_step_input),
)

if df_plot is None:
    st.error("No valid CSV files found. Check folder path and filename pattern: Gibbs_<T>K.csv")
    st.stop()

if missing_files:
    with st.expander("Missing files"):
        for f in missing_files:
            st.write(f)

st.success(f"Loaded data shape: {df_plot.shape}")

T_list = sorted(df_plot["T"].unique())


# ============================================================
# Variables
# ============================================================
variables = [
    "Co",
    "Cr",
    "Fe",
    "Ni",
    "T",
    "G_LIQ",
    "G_FCC",
    "DeltaG",
]

axis_labels = {
    "Co": "Co",
    "Cr": "Cr",
    "Fe": "Fe",
    "Ni": "Ni",
    "T": "T [K]",
    "G_LIQ": "G_LIQ [kJ/mol]",
    "G_FCC": "G_FCC [kJ/mol]",
    "DeltaG": "DeltaG [kJ/mol]",
}


# ============================================================
# Tabs
# ============================================================
tab1, tab2 = st.tabs(
    [
        "8D Parallel Plot",
        "Interpolated Gibbs Tensor + SH Explorer",
    ]
)


# ============================================================
# TAB 1: 8D Parallel Plot
# ============================================================
with tab1:
    st.subheader("Interactive 8D Parallel Plot")

    if len(df_plot) > max_rows:
        df_parallel = df_plot.sample(
            max_rows,
            random_state=int(random_seed),
        ).copy()
    else:
        df_parallel = df_plot.copy()

    df_parallel = df_parallel.reset_index(drop=True)
    df_parallel["Line_ID"] = df_parallel.index

    df_norm = df_parallel[variables].copy()

    axis_limits = {}

    for col in variables:
        col_min = df_parallel[col].min()
        col_max = df_parallel[col].max()

        axis_limits[col] = {
            "min": col_min,
            "max": col_max,
        }

        if np.isclose(col_max, col_min):
            df_norm[col] = 0.5
        else:
            df_norm[col] = (df_parallel[col] - col_min) / (col_max - col_min)

    def make_hover_text(row):
        return (
            f"<b>Line ID:</b> {int(row['Line_ID'])}<br>"
            f"<b>Co:</b> {row['Co']:.6f}<br>"
            f"<b>Cr:</b> {row['Cr']:.6f}<br>"
            f"<b>Fe:</b> {row['Fe']:.6f}<br>"
            f"<b>Ni:</b> {row['Ni']:.6f}<br>"
            f"<b>T:</b> {row['T']:.0f} K<br>"
            f"<b>G_LIQ:</b> {row['G_LIQ'] / 1000:.6f} kJ/mol<br>"
            f"<b>G_FCC:</b> {row['G_FCC'] / 1000:.6f} kJ/mol<br>"
            f"<b>DeltaG:</b> {row['DeltaG'] / 1000:.6f} kJ/mol<br>"
            f"<b>G_stable:</b> {row['G_stable'] / 1000:.6f} kJ/mol<br>"
            f"<b>Stable phase:</b> {row['Stable_Phase']}"
        )

    df_parallel["hover_text"] = df_parallel.apply(make_hover_text, axis=1)

    x_positions = {
        "Co": 0,
        "Cr": 1,
        "Fe": 2,
        "Ni": 3,
        "T": 4,
        "G_LIQ": 6,
        "G_FCC": 8,
        "DeltaG": 10,
    }

    x_vals = np.array([x_positions[v] for v in variables])

    T_min = df_parallel["T"].min()
    T_max = df_parallel["T"].max()

    fig = go.Figure()

    for idx, row in df_norm.iterrows():
        T_value = df_parallel.loc[idx, "T"]

        if np.isclose(T_max, T_min):
            T_norm = 0.5
        else:
            T_norm = (T_value - T_min) / (T_max - T_min)

        color = sample_colorscale("PiYG", 1.0 - T_norm)[0]

        y_vals = np.array([row[v] for v in variables])

        x_hover = []
        y_hover = []

        n_hover_points_per_segment = 12

        for j in range(len(x_vals) - 1):
            xs = np.linspace(
                x_vals[j],
                x_vals[j + 1],
                n_hover_points_per_segment,
            )

            ys = np.linspace(
                y_vals[j],
                y_vals[j + 1],
                n_hover_points_per_segment,
            )

            x_hover.extend(xs)
            y_hover.extend(ys)

        fig.add_trace(
            go.Scatter(
                x=x_hover,
                y=y_hover,
                mode="lines+markers",
                line=dict(
                    color=color,
                    width=line_width,
                ),
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.01,
                ),
                opacity=line_opacity,
                hovertemplate=df_parallel.loc[idx, "hover_text"] + "<extra></extra>",
                customdata=[idx] * len(x_hover),
                showlegend=False,
                name=f"Line {idx}",
            )
        )

    for var in variables:
        x = x_positions[var]

        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            line=dict(
                color="black",
                width=4,
            ),
        )

        label_color = "black"

        if var == "G_LIQ":
            label_color = "#d01c8b"
        elif var == "G_FCC":
            label_color = "#4dac26"
        elif var == "DeltaG":
            label_color = "brown"

        fig.add_annotation(
            x=x,
            y=1.08,
            text=f"<b>{axis_labels[var]}</b>",
            showarrow=False,
            font=dict(size=20, color=label_color),
        )

    for y in [0, 0.25, 0.50, 0.75, 1.0]:
        fig.add_shape(
            type="line",
            x0=-0.15,
            x1=3,
            y0=y,
            y1=y,
            line=dict(
                color="black",
                width=1.2,
                dash="dash",
            ),
        )

    for var in variables:
        x = x_positions[var]
        min_val = axis_limits[var]["min"]
        max_val = axis_limits[var]["max"]

        if var in ["G_LIQ", "G_FCC", "DeltaG"]:
            min_text = f"{min_val / 1000:.2f} kJ/mol"
            max_text = f"{max_val / 1000:.2f} kJ/mol"
        elif var == "T":
            min_text = f"{min_val:.0f} K"
            max_text = f"{max_val:.0f} K"
        else:
            min_text = f"{min_val:.3f}"
            max_text = f"{max_val:.3f}"

        fig.add_annotation(
            x=x,
            y=-0.055,
            text=f"<b>{min_text}</b>",
            showarrow=False,
            font=dict(size=13),
        )

        fig.add_annotation(
            x=x,
            y=1.035,
            text=f"<b>{max_text}</b>",
            showarrow=False,
            font=dict(size=13),
        )

    for label, y in zip(
        ["0.00", "0.10", "0.20", "0.30", "0.40"],
        [0, 0.25, 0.50, 0.75, 1.0],
    ):
        fig.add_annotation(
            x=-0.40,
            y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=15),
        )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale="PiYG",
                reversescale=True,
                cmin=T_min,
                cmax=T_max,
                color=[T_min, T_max],
                showscale=True,
                colorbar=dict(
                    title=dict(text="<b>T [K]</b>", side="right"),
                    tickmode="array",
                    tickvals=[
                        T_min,
                        (T_min + T_max) / 3,
                        2 * (T_min + T_max) / 3,
                        T_max,
                    ],
                    ticktext=[
                        f"{T_min:.0f}",
                        f"{(T_min + T_max) / 3:.0f}",
                        f"{2 * (T_min + T_max) / 3:.0f}",
                        f"{T_max:.0f}",
                    ],
                    thickness=25,
                    len=0.75,
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=760,
        margin=dict(l=40, r=80, t=80, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        xaxis=dict(
            range=[-0.7, 10.6],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=[-0.08, 1.15],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
    )

    clicked_points = plotly_events(
        fig,
        hover_event=False,
        click_event=True,
        select_event=False,
        override_height=760,
        override_width="100%",
    )

    st.subheader("Clicked Strand Values")

    if clicked_points:
        curve_number = clicked_points[0]["curveNumber"]

        if curve_number < len(df_parallel):
            selected_row = df_parallel.iloc[curve_number]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Co", f"{selected_row['Co']:.6f}")
            col2.metric("Cr", f"{selected_row['Cr']:.6f}")
            col3.metric("Fe", f"{selected_row['Fe']:.6f}")
            col4.metric("Ni", f"{selected_row['Ni']:.6f}")

            col5, col6, col7, col8 = st.columns(4)

            col5.metric("T", f"{selected_row['T']:.0f} K")
            col6.metric("G_LIQ", f"{selected_row['G_LIQ'] / 1000:.6f} kJ/mol")
            col7.metric("G_FCC", f"{selected_row['G_FCC'] / 1000:.6f} kJ/mol")
            col8.metric("DeltaG", f"{selected_row['DeltaG'] / 1000:.6f} kJ/mol")

            col9, col10 = st.columns(2)

            col9.metric("Stable G", f"{selected_row['G_stable'] / 1000:.6f} kJ/mol")
            col10.metric("Stable Phase", selected_row["Stable_Phase"])

            st.dataframe(
                pd.DataFrame(
                    [
                        selected_row[
                            [
                                "Co",
                                "Cr",
                                "Fe",
                                "Ni",
                                "T",
                                "G_LIQ",
                                "G_FCC",
                                "DeltaG",
                                "G_stable",
                                "Stable_Phase",
                            ]
                        ]
                    ]
                ),
                width='stretch',
            )

        else:
            st.info("Click a colored thermodynamic strand to display values.")

    else:
        st.info(
            "Hover shows the thermodynamic state near the mouse. "
            "Click a thin colored strand to print its values here."
        )

    if show_raw_data:
        st.subheader("Raw Loaded Data")
        st.dataframe(df_parallel, width='stretch')

    if show_normalized_data:
        st.subheader("Normalized Plot Data")
        st.dataframe(df_norm, width='stretch')


# ============================================================
# TAB 2: Interpolated Gibbs Tensor + SH Explorer
# ============================================================
with tab2:
    st.subheader("Interpolated Gibbs-Energy Tensor in Co-Cr-Fe-Ni Composition Space")

    with st.sidebar:
        st.header("🔷 TAB 2 Tensor Controls")

        st.subheader("📍 Query Point")

        q_co = st.number_input(
            "x_Co",
            0.0,
            1.0,
            0.25,
            0.01,
            format="%.2f",
            key="q_co_tab2",
        )

        q_cr = st.number_input(
            "x_Cr",
            0.0,
            1.0,
            0.25,
            0.01,
            format="%.2f",
            key="q_cr_tab2",
        )

        q_fe = st.number_input(
            "x_Fe",
            0.0,
            1.0,
            0.25,
            0.01,
            format="%.2f",
            key="q_fe_tab2",
        )

        q_t = st.selectbox(
            "Query T [K]",
            T_list,
            index=len(T_list) // 2,
            key="q_t_tab2",
        )

        comp_sum = q_co + q_cr + q_fe

        if comp_sum > 1.0:
            st.warning(f"Composition sum = {comp_sum:.2f} > 1.0")

        eval_query = st.button(
            "Evaluate Query Point",
            width='stretch',
            key="eval_btn_tab2",
        )

        st.divider()

        st.subheader("🌡️ Tensor Field")

        T_val = st.select_slider(
            "Field T [K]",
            options=T_list,
            value=T_list[len(T_list) // 2],
            key="T_viz_tab2",
        )

        grid_res = st.slider(
            "Grid Resolution",
            min_value=15,
            max_value=500,
            value=25,
            step=5,
            key="grid_res_tab2",
        )

        st.divider()

        st.subheader("🌐 Coordinates")

        coord_sys = st.radio(
            "Coordinate System",
            [
                "Cartesian (x_Co, x_Cr, x_Fe)",
                "Spherical (r, θ, φ)",
            ],
            index=0,
            key="coord_radio_tab2",
        )

        st.divider()

        st.subheader("🎨 Phase Display")

        show_phase = st.radio(
            "Mode",
            [
                "Stable Phase (Min G)",
                "LIQUID Only",
                "FCC Only",
                "Both Overlay",
            ],
            index=0,
            key="phase_mode_tab2",
        )

        cmap = st.selectbox(
            "Colormap",
            COLORMAPS,
            index=COLORMAPS.index("Viridis") if "Viridis" in COLORMAPS else 0,
            key="cmap_select_tab2",
        )

        col_s1, col_s2 = st.columns(2)

        marker_size = col_s1.slider(
            "Marker Size",
            1,
            10,
            3,
            key="mkr_size_tab2",
        )

        opacity = col_s2.slider(
            "Opacity",
            0.1,
            1.0,
            0.75,
            0.05,
            key="opacity_tab2",
        )

        st.divider()

        st.subheader("🔷 e3nn / Geometry")

        symbol = st.selectbox(
            "Marker Symbol",
            SYMBOLS,
            index=1,
            key="symbol_select_tab2",
        )

        scale_by_g = st.toggle(
            "Scale marker size by |G|",
            value=False,
            key="scale_g_tab2",
        )

        show_ref_sphere = st.toggle(
            "Show Reference Sphere",
            value=False,
            key="ref_sphere_tab2",
        )

        show_axes = st.toggle(
            "Show Coordinate Axes",
            value=False,
            key="show_axes_tab2",
        )

        show_simplex = st.toggle(
            "Show Composition Simplex",
            value=False,
            key="show_simplex_tab2",
        )

        st.divider()

        st.subheader("🌀 Spherical Harmonics")

        sh_enabled = st.toggle(
            "Enable SH Analysis",
            value=False,
            key="sh_toggle_tab2",
        )

        if sh_enabled:
            l_max = st.slider(
                "Max Degree l_max",
                1,
                8,
                4,
                key="l_max_slider_tab2",
            )

            sh_r_fixed = st.number_input(
                "Analysis Radius r",
                0.3,
                1.0,
                0.6,
                0.05,
                key="sh_r_tab2",
            )

            sh_viz_mode = st.radio(
                "SH View",
                [
                    "Coefficients",
                    "Reconstructed Surface",
                    "Error Map",
                    "Gradient Field",
                ],
                index=0,
                key="sh_viz_tab2",
            )

            enable_edit = st.toggle(
                "Edit Coefficients",
                value=False,
                key="sh_edit_tab2",
            )

            if enable_edit:
                edit_factor = st.slider(
                    "Edit Multiplier",
                    0.5,
                    2.0,
                    1.0,
                    0.1,
                    key="edit_mult_tab2",
                )
            else:
                edit_factor = 1.0

            detect_boundaries = st.toggle(
                "Detect Phase Boundaries",
                value=False,
                key="detect_boundary_tab2",
            )
        else:
            l_max = 4
            sh_r_fixed = 0.6
            sh_viz_mode = "Coefficients"
            enable_edit = False
            edit_factor = 1.0
            detect_boundaries = False

        st.divider()

        st.subheader("📦 Tensor Operations")

        export_tensor = st.button(
            "Compute Full Tensor G",
            key="compute_tensor_tab2",
        )

        export_phase_matrix = st.button(
            "Export Phase Matrix",
            key="export_phase_tab2",
        )

        st.divider()

        st.subheader("✏️ Layout")

        template = st.selectbox(
            "Template",
            [
                "plotly_white",
                "plotly_dark",
                "seaborn",
                "none",
            ],
            index=0,
            key="template_tab2",
        )

        show_grid = st.toggle(
            "Show Grid",
            value=True,
            key="show_grid_tab2",
        )

    # --------------------------------------------------------
    # Query evaluation
    # --------------------------------------------------------
    query_result = None

    if eval_query:
        if comp_sum <= 1.0:
            interp_liq_q, interp_fcc_q = build_interpolators_for_T(df_plot, q_t)

            if interp_liq_q is not None and interp_fcc_q is not None:
                pt = np.array([[q_co, q_cr, q_fe]])

                g_liq_arr = interp_liq_q(pt)
                g_fcc_arr = interp_fcc_q(pt)

                g_liq_q = float(g_liq_arr[0])
                g_fcc_q = float(g_fcc_arr[0])

                if not (np.isnan(g_liq_q) or np.isnan(g_fcc_q)):
                    g_stable_q = min(g_liq_q, g_fcc_q)
                    phase_q = "LIQUID" if g_liq_q <= g_fcc_q else "FCC"

                    query_result = {
                        "T": q_t,
                        "Co": q_co,
                        "Cr": q_cr,
                        "Fe": q_fe,
                        "Ni": round(1.0 - comp_sum, 4),
                        "G_LIQ": g_liq_q,
                        "G_FCC": g_fcc_q,
                        "G_stable": g_stable_q,
                        "Phase": phase_q,
                    }

    if query_result:
        st.success(f"Query evaluated at T = {query_result['T']} K")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("G_LIQ", f"{query_result['G_LIQ']:,.0f} J/mol")
        c2.metric("G_FCC", f"{query_result['G_FCC']:,.0f} J/mol")
        c3.metric("G_stable", f"{query_result['G_stable']:,.0f} J/mol")
        c4.metric("Phase", query_result["Phase"])
        c5.metric("x_Ni", f"{query_result['Ni']:.3f}")

        st.divider()

    # --------------------------------------------------------
    # Main interpolated grid
    # --------------------------------------------------------
    df_interp = build_interpolated_grid(
        df_plot,
        T_val,
        grid_res,
    )

    if df_interp is None or df_interp.empty:
        st.error(
            f"No interpolated data available for T = {T_val} K. "
            "Try a lower grid resolution or another temperature."
        )
        st.stop()

    pts = df_interp[["Co", "Cr", "Fe"]].values
    G_liq = df_interp["G_LIQ"].values
    G_fcc = df_interp["G_FCC"].values
    G_stable = df_interp["G_stable"].values
    stable_label = df_interp["Stable_Phase"].values

    if coord_sys == "Spherical (r, θ, φ)":
        r_data, theta_data, phi_data = cartesian_to_spherical(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
        )

        x_data = r_data
        y_data = theta_data
        z_data = phi_data

        x_title = "r"
        y_title = "θ [rad]"
        z_title = "φ [rad]"

    else:
        x_data = pts[:, 0]
        y_data = pts[:, 1]
        z_data = pts[:, 2]

        x_title = "x<sub>Co</sub>"
        y_title = "x<sub>Cr</sub>"
        z_title = "x<sub>Fe</sub>"

    sizes = np.full(len(G_stable), marker_size, dtype=float)

    if scale_by_g:
        g_norm = np.abs(G_stable)
        g_min = g_norm.min()
        g_max = g_norm.max()

        if g_max > g_min:
            sizes = 2 + 8 * (g_norm - g_min) / (g_max - g_min)

    fig3d = go.Figure()

    def make_cbar(title_text):
        return dict(
            title=dict(text=title_text),
            thickness=20,
            len=0.7,
            outlinecolor="black",
            outlinewidth=1,
        )

    marker_cfg = dict(
        symbol=symbol,
        colorscale=cmap,
        opacity=opacity,
        line=dict(width=1, color="#000000"),
    )

    custom_data = np.column_stack(
        [
            df_interp["Co"],
            df_interp["Cr"],
            df_interp["Fe"],
            df_interp["Ni"],
            df_interp["T"],
            df_interp["G_LIQ"] / 1000,
            df_interp["G_FCC"] / 1000,
            df_interp["DeltaG"] / 1000,
            df_interp["G_stable"] / 1000,
            df_interp["Stable_Phase"],
        ]
    )

    hover_template = (
        "Co: %{customdata[0]:.4f}<br>"
        "Cr: %{customdata[1]:.4f}<br>"
        "Fe: %{customdata[2]:.4f}<br>"
        "Ni: %{customdata[3]:.4f}<br>"
        "T: %{customdata[4]:.0f} K<br>"
        "G_LIQ: %{customdata[5]:.4f} kJ/mol<br>"
        "G_FCC: %{customdata[6]:.4f} kJ/mol<br>"
        "DeltaG: %{customdata[7]:.4f} kJ/mol<br>"
        "G_stable: %{customdata[8]:.4f} kJ/mol<br>"
        "Stable phase: %{customdata[9]}<extra></extra>"
    )

    if show_phase == "Stable Phase (Min G)":
        fig3d.add_trace(
            go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode="markers",
                marker=dict(
                    **marker_cfg,
                    color=G_stable,
                    size=sizes,
                    colorbar=make_cbar("G stable<br>[J/mol]"),
                ),
                name="Stable",
                customdata=custom_data,
                hovertemplate=hover_template,
            )
        )

    elif show_phase == "LIQUID Only":
        fig3d.add_trace(
            go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode="markers",
                marker=dict(
                    **marker_cfg,
                    color=G_liq,
                    size=sizes,
                    colorbar=make_cbar("G_LIQ<br>[J/mol]"),
                ),
                name="LIQUID",
                customdata=custom_data,
                hovertemplate=hover_template,
            )
        )

    elif show_phase == "FCC Only":
        fig3d.add_trace(
            go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode="markers",
                marker=dict(
                    **marker_cfg,
                    color=G_fcc,
                    size=sizes,
                    colorbar=make_cbar("G_FCC<br>[J/mol]"),
                ),
                name="FCC",
                customdata=custom_data,
                hovertemplate=hover_template,
            )
        )

    else:
        fig3d.add_trace(
            go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode="markers",
                marker=dict(
                    **marker_cfg,
                    color=G_liq,
                    size=sizes,
                    colorbar=make_cbar("G_LIQ<br>[J/mol]"),
                    opacity=opacity * 0.7,
                ),
                name="LIQUID",
                customdata=custom_data,
                hovertemplate=hover_template,
            )
        )

        fig3d.add_trace(
            go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode="markers",
                marker=dict(
                    **marker_cfg,
                    color=G_fcc,
                    size=sizes,
                    opacity=opacity * 0.7,
                ),
                name="FCC",
                customdata=custom_data,
                hovertemplate=hover_template,
            )
        )

    if query_result:
        if coord_sys == "Spherical (r, θ, φ)":
            x_q, y_q, z_q = cartesian_to_spherical(
                np.array([query_result["Co"]]),
                np.array([query_result["Cr"]]),
                np.array([query_result["Fe"]]),
            )
        else:
            x_q = [query_result["Co"]]
            y_q = [query_result["Cr"]]
            z_q = [query_result["Fe"]]

        fig3d.add_trace(
            go.Scatter3d(
                x=x_q,
                y=y_q,
                z=z_q,
                mode="markers",
                marker=dict(
                    size=14,
                    color="red",
                    symbol="diamond",
                    line=dict(width=2, color="white"),
                ),
                name="Query",
                hovertemplate=(
                    f"<b>QUERY</b><br>"
                    f"T = {query_result['T']} K<br>"
                    f"x_Co = {query_result['Co']:.3f}<br>"
                    f"x_Cr = {query_result['Cr']:.3f}<br>"
                    f"x_Fe = {query_result['Fe']:.3f}<br>"
                    f"x_Ni = {query_result['Ni']:.3f}<br>"
                    f"G = {query_result['G_stable']:,.0f} J/mol<br>"
                    f"Phase = {query_result['Phase']}<extra></extra>"
                ),
            )
        )

    if show_ref_sphere:
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)

        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))

        fig3d.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                opacity=0.05,
                colorscale=[[0, "gray"], [1, "gray"]],
                showscale=False,
                hoverinfo="skip",
                name="Reference Sphere",
            )
        )

    if show_axes:
        for ax, color, label in [
            ([1.1, 0, 0], "red", "Co"),
            ([0, 1.1, 0], "green", "Cr"),
            ([0, 0, 1.1], "blue", "Fe"),
        ]:
            fig3d.add_trace(
                go.Scatter3d(
                    x=[0, ax[0]],
                    y=[0, ax[1]],
                    z=[0, ax[2]],
                    mode="lines+text",
                    line=dict(color=color, width=3),
                    text=["", label],
                    textfont=dict(size=11, color=color),
                    hoverinfo="skip",
                    name=f"{label} axis",
                )
            )

    if show_simplex:
        edges = [
            [(1, 0, 0), (0, 1, 0)],
            [(1, 0, 0), (0, 0, 1)],
            [(1, 0, 0), (0, 0, 0)],
            [(0, 1, 0), (0, 0, 1)],
            [(0, 1, 0), (0, 0, 0)],
            [(0, 0, 1), (0, 0, 0)],
        ]

        ex = []
        ey = []
        ez = []

        for e in edges:
            ex += [e[0][0], e[1][0], None]
            ey += [e[0][1], e[1][1], None]
            ez += [e[0][2], e[1][2], None]

        fig3d.add_trace(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(
                    color="black",
                    width=1.5,
                    dash="dash",
                ),
                name="Composition Simplex",
                hoverinfo="skip",
            )
        )

    axis_cfg = dict(
        showbackground=True,
        backgroundcolor="#ffffff" if template != "plotly_dark" else "#0e1117",
        gridcolor="rgba(128,128,128,0.3)" if show_grid else "rgba(0,0,0,0)",
        showgrid=show_grid,
        zerolinecolor="rgba(128,128,128,0.4)" if show_grid else "rgba(0,0,0,0)",
    )

    fig3d.update_layout(
        template=template if template != "none" else None,
        scene=dict(
            xaxis=dict(title=x_title, **axis_cfg),
            yaxis=dict(title=y_title, **axis_cfg),
            zaxis=dict(title=z_title, **axis_cfg),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
            ),
        ),
        title=dict(
            text=f"Gibbs Energy Tensor at T = {T_val} K | Points: {len(df_interp):,}",
            font=dict(size=16),
        ),
        height=780,
        margin=dict(l=0, r=0, b=0, t=45),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    clicked_3d = plotly_events(
        fig3d,
        hover_event=False,
        click_event=True,
        select_event=False,
        override_height=780,
        override_width="100%",
    )

    st.subheader("Clicked Interpolated State")

    if clicked_3d:
        point_id = clicked_3d[0]["pointNumber"]

        if point_id < len(df_interp):
            selected_row = df_interp.iloc[point_id]

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Co", f"{selected_row['Co']:.5f}")
            c2.metric("Cr", f"{selected_row['Cr']:.5f}")
            c3.metric("Fe", f"{selected_row['Fe']:.5f}")
            c4.metric("Ni", f"{selected_row['Ni']:.5f}")

            c5, c6, c7, c8 = st.columns(4)

            c5.metric("T", f"{selected_row['T']:.0f} K")
            c6.metric("G_LIQ", f"{selected_row['G_LIQ'] / 1000:.5f} kJ/mol")
            c7.metric("G_FCC", f"{selected_row['G_FCC'] / 1000:.5f} kJ/mol")
            c8.metric("DeltaG", f"{selected_row['DeltaG'] / 1000:.5f} kJ/mol")

            c9, c10 = st.columns(2)

            c9.metric("Stable G", f"{selected_row['G_stable'] / 1000:.5f} kJ/mol")
            c10.metric("Stable Phase", selected_row["Stable_Phase"])

            st.dataframe(
                pd.DataFrame([selected_row]),
                width='stretch',
            )
        else:
            st.info("Click an interpolated point to display values.")

    else:
        st.info(
            "Hover shows the thermodynamic state near the mouse. "
            "Click an interpolated point to print its values here."
        )

    # --------------------------------------------------------
    # Spherical harmonics analysis
    # --------------------------------------------------------
    if sh_enabled:
        st.divider()
        st.subheader("🌀 Spherical Harmonics Analysis")

        r_vals, theta_vals, phi_vals = cartesian_to_spherical(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
        )

        mask_r = np.abs(r_vals - sh_r_fixed) < 0.05

        pts_sh = pts[mask_r]
        G_sh_vals = G_stable[mask_r]
        theta_sh = theta_vals[mask_r]
        phi_sh = phi_vals[mask_r]

        required_points = (l_max + 1) ** 2

        if len(pts_sh) > required_points:
            with st.spinner("Fitting spherical harmonics..."):
                coeffs, G_recon, rmse, cond_num = fit_spherical_harmonics(
                    G_sh_vals,
                    theta_sh,
                    phi_sh,
                    l_max,
                    reg_lambda=1e-5,
                )

            coeffs_display = coeffs.copy()

            if enable_edit:
                coeff_labels = [
                    f"Y_{l}^{m}"
                    for l in range(l_max + 1)
                    for m in range(-l, l + 1)
                ]

                coeff_select = st.multiselect(
                    "Select coefficients to edit",
                    options=coeff_labels,
                    default=[
                        f"Y_{l}^{0}"
                        for l in range(min(3, l_max + 1))
                    ],
                    key="coeff_edit_select_tab2",
                )

                idx = 0

                for l in range(l_max + 1):
                    for m in range(-l, l + 1):
                        label = f"Y_{l}^{m}"

                        if label in coeff_select:
                            coeffs_display[idx] *= edit_factor

                        idx += 1

            denominator = np.sum((G_sh_vals - np.mean(G_sh_vals)) ** 2)
            numerator = np.sum((G_sh_vals - G_recon) ** 2)

            if denominator > 0:
                r2 = 1 - numerator / denominator
            else:
                r2 = np.nan

            st.success(
                f"SH Fit: RMSE = {rmse:.1f} J/mol | "
                f"R² = {r2:.4f} | Cond# = {cond_num:.1e}"
            )

            if sh_viz_mode == "Coefficients":
                coeff_labels = [
                    f"Y_{l}^{m}"
                    for l in range(l_max + 1)
                    for m in range(-l, l + 1)
                ]

                fig_coeffs = go.Figure(
                    data=[
                        go.Bar(
                            x=coeff_labels,
                            y=coeffs_display,
                            marker=dict(
                                color=coeffs_display,
                                colorscale="RdBu",
                                showscale=True,
                            ),
                        )
                    ]
                )

                fig_coeffs.update_layout(
                    title=f"SH Coefficients, l_max = {l_max}",
                    xaxis_title="Spherical Harmonic Y_lm",
                    yaxis_title="Coefficient [J/mol]",
                    xaxis_tickangle=-45,
                    height=350,
                    margin=dict(t=40, b=80),
                )

                st.plotly_chart(fig_coeffs, width='stretch')

                with st.expander("Coefficient Table"):
                    coeff_df = pd.DataFrame(
                        {
                            "l": [
                                l
                                for l in range(l_max + 1)
                                for _ in range(2 * l + 1)
                            ],
                            "m": [
                                m
                                for l in range(l_max + 1)
                                for m in range(-l, l + 1)
                            ],
                            "coefficient": coeffs_display,
                            "label": coeff_labels,
                        }
                    )

                    st.dataframe(
                        coeff_df.style.format({"coefficient": "{:.3f}"}),
                        height=300,
                    )

            elif sh_viz_mode == "Reconstructed Surface":
                X_sh, Y_sh, Z_sh, G_sh_surface, th_grid, ph_grid = generate_sh_surface(
                    coeffs_display,
                    l_max,
                    r_base=sh_r_fixed,
                    r_scale=0.25,
                )

                fig_sh = go.Figure(
                    data=[
                        go.Surface(
                            x=X_sh,
                            y=Y_sh,
                            z=Z_sh,
                            surfacecolor=G_sh_surface,
                            colorscale=cmap,
                            opacity=0.95,
                            colorbar=dict(title="G [J/mol]"),
                        )
                    ]
                )

                fig_sh.update_layout(
                    title=f"SH-Reconstructed G at r = {sh_r_fixed:.2f}",
                    scene=dict(
                        xaxis_title="x",
                        yaxis_title="y",
                        zaxis_title="z",
                        aspectmode="cube",
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                )

                st.plotly_chart(fig_sh, width='stretch')

            elif sh_viz_mode == "Error Map":
                residual = G_sh_vals - G_recon

                x_err, y_err, z_err = spherical_to_cartesian(
                    np.full_like(theta_sh, sh_r_fixed),
                    theta_sh,
                    phi_sh,
                )

                fig_err = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=x_err,
                            y=y_err,
                            z=z_err,
                            mode="markers",
                            marker=dict(
                                size=4,
                                color=residual,
                                colorscale="RdBu_r",
                                colorbar=dict(title="Residual"),
                                opacity=0.85,
                            ),
                        )
                    ]
                )

                fig_err.update_layout(
                    title=f"Reconstruction Error at r = {sh_r_fixed:.2f}",
                    scene=dict(aspectmode="cube"),
                )

                st.plotly_chart(fig_err, width='stretch')

            elif sh_viz_mode == "Gradient Field":
                dG_dth, dG_dph = compute_sh_gradient(
                    coeffs_display,
                    l_max,
                    theta_sh,
                    phi_sh,
                )

                grad_mag = np.sqrt(dG_dth**2 + dG_dph**2)

                x_grad, y_grad, z_grad = spherical_to_cartesian(
                    np.full_like(theta_sh, sh_r_fixed),
                    theta_sh,
                    phi_sh,
                )

                fig_grad = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=x_grad,
                            y=y_grad,
                            z=z_grad,
                            mode="markers",
                            marker=dict(
                                size=5,
                                color=grad_mag,
                                colorscale="Viridis",
                                colorbar=dict(title="|∇G|"),
                                opacity=0.9,
                            ),
                        )
                    ]
                )

                fig_grad.update_layout(
                    title=f"SH Gradient Magnitude |∇G| at r = {sh_r_fixed:.2f}",
                    scene=dict(aspectmode="cube"),
                )

                st.plotly_chart(fig_grad, width='stretch')

            with st.expander("Export SH Coefficients"):
                coeff_export_df = pd.DataFrame(
                    {
                        "l": [
                            l
                            for l in range(l_max + 1)
                            for _ in range(2 * l + 1)
                        ],
                        "m": [
                            m
                            for l in range(l_max + 1)
                            for m in range(-l, l + 1)
                        ],
                        "coefficient": coeffs_display,
                    }
                )

                export_data = {
                    "l_max": l_max,
                    "r_reference": float(sh_r_fixed),
                    "temperature_K": int(T_val),
                    "coefficients": coeffs_display.tolist(),
                    "rmse_j_mol": float(rmse),
                    "r_squared": float(r2),
                    "n_data_points": int(len(pts_sh)),
                    "edited": bool(enable_edit),
                    "edit_factor": float(edit_factor) if enable_edit else None,
                }

                st.download_button(
                    "Download SH Coefficients CSV",
                    data=coeff_export_df.to_csv(index=False),
                    file_name=f"SH_coeffs_T{T_val}K_r{sh_r_fixed:.2f}.csv",
                    mime="text/csv",
                )

                st.download_button(
                    "Download SH Coefficients JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"SH_coeffs_T{T_val}K_r{sh_r_fixed:.2f}.json",
                    mime="application/json",
                )

            if detect_boundaries:
                st.subheader("Phase Boundary Detection via SH Gradient")

                dG_dth, dG_dph = compute_sh_gradient(
                    coeffs_display,
                    l_max,
                    theta_sh,
                    phi_sh,
                )

                grad_mag = np.sqrt(dG_dth**2 + dG_dph**2)

                boundary_threshold = np.percentile(grad_mag, 85)
                boundary_mask = grad_mag > boundary_threshold

                if np.any(boundary_mask):
                    x_bnd, y_bnd, z_bnd = spherical_to_cartesian(
                        np.full_like(theta_sh[boundary_mask], sh_r_fixed),
                        theta_sh[boundary_mask],
                        phi_sh[boundary_mask],
                    )

                    fig_bnd = go.Figure()

                    fig_bnd.add_trace(
                        go.Scatter3d(
                            x=x_data,
                            y=y_data,
                            z=z_data,
                            mode="markers",
                            marker=dict(
                                size=2,
                                color="lightgray",
                                opacity=0.3,
                            ),
                            name="All points",
                            hoverinfo="skip",
                        )
                    )

                    fig_bnd.add_trace(
                        go.Scatter3d(
                            x=x_bnd,
                            y=y_bnd,
                            z=z_bnd,
                            mode="markers",
                            marker=dict(
                                size=6,
                                color="red",
                                symbol="diamond",
                            ),
                            name="Phase Boundary",
                            hovertemplate=(
                                "Boundary Point<br>"
                                "x=%{x:.3f}<br>"
                                "y=%{y:.3f}<br>"
                                "z=%{z:.3f}<extra></extra>"
                            ),
                        )
                    )

                    fig_bnd.update_layout(
                        title=f"Detected Phase Boundaries at T = {T_val} K, r = {sh_r_fixed:.2f}",
                        scene=dict(aspectmode="cube"),
                    )

                    st.plotly_chart(fig_bnd, width='stretch')

                    bnd_df = pd.DataFrame(
                        {
                            "x_Co": x_bnd,
                            "x_Cr": y_bnd,
                            "x_Fe": z_bnd,
                            "theta_rad": theta_sh[boundary_mask],
                            "phi_rad": phi_sh[boundary_mask],
                            "gradient_magnitude": grad_mag[boundary_mask],
                        }
                    )

                    st.download_button(
                        "Download Boundary CSV",
                        data=bnd_df.to_csv(index=False),
                        file_name=f"phase_boundary_T{T_val}K_r{sh_r_fixed:.2f}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No clear phase boundary detected at this radius.")

        else:
            st.warning(
                f"Need more than {required_points} points at r ≈ {sh_r_fixed:.2f}; "
                f"found {len(pts_sh)}. Try another radius or increase grid resolution."
            )

    # --------------------------------------------------------
    # Tensor export
    # --------------------------------------------------------
    if export_tensor or export_phase_matrix:
        st.divider()
        st.subheader("📦 Tensor Calculations")

        with st.spinner("Computing full tensor G[x_Co, x_Cr, x_Fe, T]..."):
            tensor_result = compute_gibbs_tensor(
                df_plot,
                comp_grid_res=15,
            )

        if export_tensor:
            st.success("Full Gibbs tensor computed.")

            G_stable_tensor = tensor_result["G_stable"]
            valid_G = G_stable_tensor[~np.isnan(G_stable_tensor)]

            if len(valid_G) > 0:
                t1, t2, t3, t4 = st.columns(4)

                t1.metric("Min G", f"{np.nanmin(valid_G):,.0f} J/mol")
                t2.metric("Max G", f"{np.nanmax(valid_G):,.0f} J/mol")
                t3.metric("Mean |G|", f"{np.mean(np.abs(valid_G)):,.0f} J/mol")
                t4.metric("Valid Points", f"{len(valid_G):,}")

            tensor_dict = {
                "G_LIQ": tensor_result["G_LIQ"].tolist(),
                "G_FCC": tensor_result["G_FCC"].tolist(),
                "G_stable": tensor_result["G_stable"].tolist(),
                "T_array": tensor_result["T_array"].tolist(),
                "comp_vals": tensor_result["comp_vals"].tolist(),
                "metadata": {
                    "shape": tensor_result["G_stable"].shape,
                    "units": "J/mol",
                    "phases": ["LIQUID", "FCC"],
                },
            }

            st.download_button(
                "Download Full Tensor JSON",
                data=json.dumps(tensor_dict, indent=2),
                file_name="Gibbs_tensor_CoCrFeNi.json",
                mime="application/json",
            )

        if export_phase_matrix:
            st.success("Phase matrix computed.")

            phase_mat = tensor_result["phase_matrix"]
            T_arr = tensor_result["T_array"]
            comp_vals = tensor_result["comp_vals"]

            t_idx = int(np.argmin(np.abs(T_arr - T_val)))
            phase_slice = phase_mat[t_idx]

            valid_phase = phase_slice[phase_slice >= 0]

            if len(valid_phase) > 0:
                liq_pct = np.sum(valid_phase == 0) / len(valid_phase) * 100
                fcc_pct = np.sum(valid_phase == 1) / len(valid_phase) * 100

                p1, p2, p3 = st.columns(3)

                p1.metric("LIQUID Region", f"{liq_pct:.1f}%")
                p2.metric("FCC Region", f"{fcc_pct:.1f}%")
                p3.metric("Valid Phase Points", f"{len(valid_phase):,}")

            fe_slice_idx = len(comp_vals) // 2
            phase_2d = phase_mat[t_idx, :, :, fe_slice_idx]

            fig_phase = go.Figure(
                data=go.Heatmap(
                    z=phase_2d,
                    x=comp_vals,
                    y=comp_vals,
                    colorscale=[
                        [0.0, "lightblue"],
                        [0.5, "lightblue"],
                        [0.5, "lightcoral"],
                        [1.0, "lightcoral"],
                    ],
                    colorbar=dict(
                        title="Phase",
                        tickvals=[0, 1],
                        ticktext=["LIQUID", "FCC"],
                    ),
                    showscale=True,
                )
            )

            fig_phase.update_layout(
                title=f"Phase Map at T = {T_arr[t_idx]} K, x_Fe ≈ {comp_vals[fe_slice_idx]:.2f}",
                xaxis_title="x_Co",
                yaxis_title="x_Cr",
                width=600,
                height=500,
            )

            st.plotly_chart(fig_phase, use_container_width=False)

            phase_export = {
                "phase_matrix": phase_mat.tolist(),
                "T_array": T_arr.tolist(),
                "comp_vals": comp_vals.tolist(),
                "phase_encoding": {
                    "0": "LIQUID",
                    "1": "FCC",
                    "-1": "Invalid or unavailable",
                },
                "metadata": {
                    "shape": phase_mat.shape,
                    "description": "0 = LIQUID stable, 1 = FCC stable, -1 = invalid/unavailable",
                },
            }

            st.download_button(
                "Download Phase Matrix JSON",
                data=json.dumps(phase_export, indent=2),
                file_name="phase_matrix_CoCrFeNi.json",
                mime="application/json",
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    st.divider()
    st.subheader("📊 Current Grid Statistics")

    stat1, stat2, stat3, stat4 = st.columns(4)

    stat1.metric("Min G stable", f"{G_stable.min():,.0f} J/mol")
    stat2.metric("Max G stable", f"{G_stable.max():,.0f} J/mol")
    stat3.metric("Mean |G stable|", f"{np.mean(np.abs(G_stable)):,.0f} J/mol")

    liq_pct = np.sum(G_liq <= G_fcc) / len(G_liq) * 100
    stat4.metric("LIQUID Fraction", f"{liq_pct:.1f}%")

    with st.expander("Interpolated dataframe"):
        st.dataframe(df_interp, width='stretch')
