"""
Standalone Streamlit app for the CoCrFeNi temperature-composition-force multi-ring chart.

Expected folder structure:

    sunburst_multiring_chart_self_contained.py
    csv_files/
        Gibbs_800K.csv
        Gibbs_1000K.csv
        Gibbs_1200K.csv
        ...

Each CSV file must contain the columns:
    Co, Cr, Fe, Ni, G_LIQ, G_FCC

Run with:
    streamlit run sunburst_multiring_chart_self_contained.py
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CoCrFeNi Multi-Ring Chart",
    layout="wide",
    page_icon="🌞",
    initial_sidebar_state="expanded"
)

st.title("🌞 CoCrFeNi Temperature–Composition–Interface Driving Force Multi-Ring Chart")


# ============================================================
# PATH AND DATA LOADING
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")

REQUIRED_COLUMNS = ["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"]


@st.cache_data(show_spinner=True)
def load_temperature_data(csv_dir: str):
    """Load all Gibbs_XXXXK.csv files from csv_dir into a dictionary keyed by temperature."""
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))

    if not files:
        return {}, [], []

    data = {}
    skipped_files = []

    for file_path in files:
        basename = Path(file_path).stem

        try:
            temperature = int(basename.replace("Gibbs_", "").replace("K", ""))
        except ValueError:
            skipped_files.append((file_path, "Could not extract temperature from filename."))
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            skipped_files.append((file_path, f"Could not read CSV: {exc}"))
            continue

        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            skipped_files.append((file_path, f"Missing columns: {missing_columns}"))
            continue

        df = df[REQUIRED_COLUMNS].copy()

        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=REQUIRED_COLUMNS)

        df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
        df = df[np.abs(df["sum_x"] - 1.0) < 1e-6].copy()

        if df.empty:
            skipped_files.append((file_path, "No valid rows after composition-sum filtering."))
            continue

        data[temperature] = df

    temperatures = sorted(data.keys())
    return data, temperatures, skipped_files


data_by_T, temperatures, skipped_files = load_temperature_data(CSV_FILES_DIR)

if not data_by_T:
    st.error(f"❌ No valid CSV files found in `{CSV_FILES_DIR}`")
    st.info("Expected files such as `csv_files/Gibbs_1000K.csv`.")
    st.info("Required columns: `Co`, `Cr`, `Fe`, `Ni`, `G_LIQ`, `G_FCC`.")

    if skipped_files:
        with st.expander("Show skipped files"):
            for file_path, reason in skipped_files:
                st.write(f"- `{file_path}`: {reason}")

    st.stop()

if skipped_files:
    with st.expander("⚠️ Some files were skipped", expanded=False):
        for file_path, reason in skipped_files:
            st.write(f"- `{file_path}`: {reason}")


# ============================================================
# CONSTANTS AND HELPER FUNCTIONS
# ============================================================

PURE_VM = {
    "Co": 6.80e-6,
    "Cr": 7.23e-6,
    "Fe": 7.09e-6,
    "Ni": 6.59e-6
}

T_MIN_NORMALIZE = 300
T_MAX_NORMALIZE = 3300
GAMMA_LIQUID_FCC = 0.6
DEFAULT_DV = 1e-18

GRAIN_SHAPE_FACTORS = {
    "Spherical (k=2)": 2.0,
    "Tetrakaidecahedron (k=3)": 3.0,
    "Equiaxed cubic (k=6)": 6.0
}


def composition_dependent_vm(x_co, x_cr, x_fe, x_ni):
    """Composition-dependent molar volume by linear mixing."""
    return (
        x_co * PURE_VM["Co"]
        + x_cr * PURE_VM["Cr"]
        + x_fe * PURE_VM["Fe"]
        + x_ni * PURE_VM["Ni"]
    )


def normalize_temperature(T):
    """Normalize temperature to approximately [0, 1] using 300–3300 K."""
    return (T - T_MIN_NORMALIZE) / (T_MAX_NORMALIZE - T_MIN_NORMALIZE)


def compute_Sv(grain_size_m, shape_factor):
    """Grain-boundary/interfacial area density: Sv = k / d."""
    return shape_factor / grain_size_m


def compute_total_area(Sv, sample_volume_m3):
    """Total interfacial area: A_total = Sv * V."""
    return Sv * sample_volume_m3


def compute_curvature_radius(grain_size_m):
    """Local curvature radius approximated as D/4."""
    return grain_size_m / 4.0


def compute_capillary_pressure(gamma, curvature_r):
    """Capillary pressure: P_capillary = 2 gamma / r."""
    return (2.0 * gamma) / curvature_r


def compute_net_pressure(P_chem, P_capillary):
    """Net pressure: P_net = P_chem - P_capillary."""
    return P_chem - P_capillary


def compute_differential_force(P_net, Sv, dV):
    """Differential force: dF_net = P_net * Sv * dV."""
    return P_net * Sv * dV


# ============================================================
# SIDEBAR SUMMARY
# ============================================================

st.sidebar.header("📊 Loaded data")
st.sidebar.metric("Temperature files", len(temperatures))
st.sidebar.metric("Temperature range", f"{min(temperatures)}–{max(temperatures)} K")
st.sidebar.metric("Rows at first T", len(data_by_T[temperatures[0]]))
st.sidebar.caption(f"CSV folder: `{CSV_FILES_DIR}`")


# ============================================================
# CHART CONTROLS
# ============================================================

st.header("Chart controls")

col_sb1, col_sb2, col_sb3, col_sb4, col_sb5 = st.columns(5)

with col_sb1:
    area_mode = st.selectbox(
        "Area mode",
        ["Grain Size Derived (Sv x V)", "Direct Input (A)"],
        index=0
    )

with col_sb2:
    temp_cmap = st.selectbox(
        "Temperature colorscale",
        [
            "thermal", "inferno", "magma", "plasma", "viridis",
            "Plotly3", "Portland", "Bluered", "cividis",
            "blackbody", "hot", "turbo", "temps"
        ],
        index=5
    )

with col_sb3:
    force_cmap = st.selectbox(
        "Force colorscale",
        [
            "Portland_r", "Portland", "Plotly3", "ice", "haline",
            "deep", "dense", "teal", "tealgrn", "blues",
            "blugrn", "pubu", "electric"
        ],
        index=0
    )

with col_sb4:
    sb_sample = st.slider(
        "Compositions per temperature",
        min_value=3,
        max_value=12,
        value=5,
        step=1
    )

with col_sb5:
    sample_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1
    )


# ============================================================
# FORCE / AREA PARAMETERS
# ============================================================

if area_mode == "Grain Size Derived (Sv x V)":
    st.subheader("Grain-size-derived force parameters")

    col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)

    with col_g1:
        grain_size_um = st.number_input(
            "Average grain size d (μm)",
            min_value=0.001,
            max_value=10000.0,
            value=2.5,
            step=0.1,
            format="%.3f"
        )

    with col_g2:
        shape_choice = st.selectbox(
            "Grain shape factor",
            list(GRAIN_SHAPE_FACTORS.keys()),
            index=1
        )

    with col_g3:
        sample_volume_cm3 = st.number_input(
            "Sample volume V (cm³)",
            min_value=1e-9,
            max_value=1e6,
            value=1.0,
            step=0.1,
            format="%.3f"
        )

    with col_g4:
        dV_um3 = st.number_input(
            "Local volume dV (μm³)",
            min_value=0.001,
            max_value=1000.0,
            value=1.0,
            step=0.1,
            format="%.3f"
        )

    with col_g5:
        gamma = st.number_input(
            "γ Liquid/FCC (N/m)",
            min_value=0.01,
            max_value=5.0,
            value=GAMMA_LIQUID_FCC,
            step=0.01,
            format="%.2f"
        )

    use_capillary = st.checkbox(
        "Enable capillary correction: P_net = P_chem − 2γ/r",
        value=True
    )

    grain_size_m = grain_size_um * 1e-6
    shape_factor = GRAIN_SHAPE_FACTORS[shape_choice]
    sample_volume_m3 = sample_volume_cm3 * 1e-6
    dV = dV_um3 * 1e-18

    Sv = compute_Sv(grain_size_m, shape_factor)
    interface_area = compute_total_area(Sv, sample_volume_m3)
    sb_area = interface_area

    c1, c2, c3 = st.columns(3)
    c1.metric("Sv", f"{Sv:.2e} m²/m³")
    c2.metric("A_total", f"{interface_area:.2e} m²")
    c3.metric("dV", f"{dV:.2e} m³")

else:
    st.subheader("Direct-area force parameters")

    grain_size_m = None
    Sv = None
    dV = DEFAULT_DV
    gamma = 0.0
    use_capillary = False

    sb_area = st.number_input(
        "Reference area A (m²)",
        min_value=1e-12,
        max_value=1e2,
        value=1e-8,
        step=1e-10,
        format="%.2e"
    )


# ============================================================
# BUILD MULTI-RING CHART
# ============================================================

st.header("🌞 Multi-Ring Mechanical Force–Thermodynamic State Space Chart")

st.info(
    "Inner ring = Temperature, middle ring = Co/Cr/Fe/Ni composition, "
    "outer ring = interface driving force."
)

with st.spinner("Building multi-ring chart..."):

    temperatures_sorted = sorted(temperatures)
    n_temp = len(temperatures_sorted)

    if n_temp == 0:
        st.warning("No temperatures available.")
        st.stop()

    # ============================================================
    # RING GEOMETRY
    # ============================================================

    temp_ring_base = 0.00
    temp_ring_thickness = 0.85

    comp_ring_base = 1.02
    comp_ring_thickness = 1.25

    force_ring_base = 2.45
    force_ring_thickness = 0.90

    temp_width = 360.0 / n_temp

    # ============================================================
    # STYLE SETTINGS
    # ============================================================

    FONT_FAMILY = "Arial Black, Arial, sans-serif"

    TITLE_FONT_SIZE = 30
    SUBTITLE_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 19
    COLORBAR_TITLE_SIZE = 21
    COLORBAR_TICK_SIZE = 17
    HOVER_FONT_SIZE = 18

    RING_BORDER_COLOR = "rgba(15,15,15,0.98)"
    RING_BORDER_WIDTH = 1.9

    COMPOSITION_BORDER_COLOR = "rgba(255,255,255,0.98)"
    COMPOSITION_BORDER_WIDTH = 1.15

    PAPER_BG = "white"
    PLOT_BG = "white"

    comp_colors_map = {
        "Co": "#0057B8",
        "Cr": "#D62828",
        "Fe": "#2A9D3F",
        "Ni": "#F4C430"
    }

    # ============================================================
    # DATA CONTAINERS
    # ============================================================

    temp_theta = []
    temp_widths = []
    temp_r = []
    temp_base = []
    temp_colors = []
    temp_custom = []

    element_data = {
        "Co": {"theta": [], "width": [], "r": [], "base": [], "custom": []},
        "Cr": {"theta": [], "width": [], "r": [], "base": [], "custom": []},
        "Fe": {"theta": [], "width": [], "r": [], "base": [], "custom": []},
        "Ni": {"theta": [], "width": [], "r": [], "base": [], "custom": []}
    }

    force_theta = []
    force_widths = []
    force_r = []
    force_base = []
    force_colors = []
    force_custom = []

    rng = np.random.default_rng(int(sample_seed))

    # ============================================================
    # BUILD DATA
    # ============================================================

    for iT, T_sun in enumerate(temperatures_sorted):

        theta_start = iT * temp_width
        theta_center_temp = theta_start + temp_width / 2.0

        # ------------------------------
        # Temperature ring
        # ------------------------------

        temp_theta.append(theta_center_temp)
        temp_widths.append(temp_width)
        temp_r.append(temp_ring_thickness)
        temp_base.append(temp_ring_base)
        temp_colors.append(float(T_sun))
        temp_custom.append([
            T_sun,
            normalize_temperature(T_sun),
            f"{T_sun} K"
        ])

        df_temp = data_by_T[T_sun].copy()

        if df_temp.empty:
            continue

        n_samples = min(int(sb_sample), len(df_temp))

        if n_samples <= 0:
            continue

        if n_samples >= len(df_temp):
            sample_df = df_temp.sample(
                frac=1.0,
                random_state=int(sample_seed)
            ).reset_index(drop=True)
        else:
            sample_idx = rng.choice(
                len(df_temp),
                size=n_samples,
                replace=False
            )
            sample_df = df_temp.iloc[sample_idx].copy().reset_index(drop=True)

        sample_df = sample_df.sort_values(
            ["Co", "Cr", "Fe", "Ni"]
        ).reset_index(drop=True)

        comp_width = temp_width / n_samples

        for j, row in sample_df.iterrows():

            theta_local_start = theta_start + j * comp_width
            theta_local_center = theta_local_start + comp_width / 2.0

            x_co_s = float(row["Co"])
            x_cr_s = float(row["Cr"])
            x_fe_s = float(row["Fe"])
            x_ni_s = float(row["Ni"])

            g_liq_s = float(row["G_LIQ"])
            g_fcc_s = float(row["G_FCC"])

            delta_G_s = g_fcc_s - g_liq_s

            V_m_s = composition_dependent_vm(
                x_co_s,
                x_cr_s,
                x_fe_s,
                x_ni_s
            )

            # Chemical driving pressure in Pa.
            # Positive P_chem drives LIQUID -> FCC growth.
            P_chem_pa_s = -delta_G_s / V_m_s
            P_chem_mpa_s = P_chem_pa_s / 1e6

            if use_capillary and grain_size_m is not None:
                chart_curvature_r = compute_curvature_radius(grain_size_m)
                chart_P_capillary = compute_capillary_pressure(
                    gamma,
                    chart_curvature_r
                )
                P_net_s = compute_net_pressure(
                    P_chem_pa_s,
                    chart_P_capillary
                )
                P_net_mpa_s = P_net_s / 1e6
            else:
                P_net_s = P_chem_pa_s
                P_net_mpa_s = P_chem_mpa_s

            if area_mode == "Grain Size Derived (Sv x V)" and Sv is not None:
                net_force_s = compute_differential_force(P_net_s, Sv, dV)
                force_mode_s = "dF_net = P_net × Sv × dV"
            else:
                net_force_s = P_net_s * float(sb_area)
                force_mode_s = "F = P_net × A"

            comp_vals = {
                "Co": x_co_s,
                "Cr": x_cr_s,
                "Fe": x_fe_s,
                "Ni": x_ni_s
            }

            cumulative_base = comp_ring_base

            # ------------------------------
            # Composition ring
            # ------------------------------

            for el in ["Co", "Cr", "Fe", "Ni"]:

                thickness_el = comp_vals[el] * comp_ring_thickness

                element_data[el]["theta"].append(theta_local_center)
                element_data[el]["width"].append(comp_width)
                element_data[el]["r"].append(thickness_el)
                element_data[el]["base"].append(cumulative_base)

                element_data[el]["custom"].append([
                    T_sun,
                    x_co_s,
                    x_cr_s,
                    x_fe_s,
                    x_ni_s,
                    delta_G_s,
                    P_chem_mpa_s,
                    net_force_s,
                    el,
                    comp_vals[el],
                    P_net_mpa_s,
                    force_mode_s
                ])

                cumulative_base += thickness_el

            # ------------------------------
            # Force ring
            # ------------------------------

            force_theta.append(theta_local_center)
            force_widths.append(comp_width)
            force_r.append(force_ring_thickness)
            force_base.append(force_ring_base)
            force_colors.append(net_force_s)

            force_custom.append([
                T_sun,
                x_co_s,
                x_cr_s,
                x_fe_s,
                x_ni_s,
                delta_G_s,
                P_chem_mpa_s,
                net_force_s,
                P_net_mpa_s,
                force_mode_s
            ])

    # ============================================================
    # COLOR SCALE LIMITS
    # ============================================================

    if len(force_colors) > 0:
        max_abs_force = max(
            abs(float(np.nanmin(force_colors))),
            abs(float(np.nanmax(force_colors)))
        )

        if max_abs_force == 0:
            max_abs_force = 1.0
    else:
        max_abs_force = 1.0

    temp_min_tick = float(min(temperatures_sorted))
    temp_max_tick = float(max(temperatures_sorted))
    temp_tickvals = np.linspace(temp_min_tick, temp_max_tick, 5)
    temp_ticktext = [f"{v:.0f}" for v in temp_tickvals]

    force_min_tick = -float(max_abs_force)
    force_max_tick = float(max_abs_force)
    force_tickvals = np.linspace(force_min_tick, force_max_tick, 5)
    force_ticktext = [f"{v:.2e}" for v in force_tickvals]

    # ============================================================
    # BUILD FIGURE
    # ============================================================

    fig = go.Figure()

    # ============================================================
    # INNER TEMPERATURE RING
    # ============================================================

    fig.add_trace(
        go.Barpolar(
            theta=temp_theta,
            width=temp_widths,
            r=temp_r,
            base=temp_base,
            marker=dict(
                color=temp_colors,
                colorscale=temp_cmap,
                cmin=min(temperatures_sorted),
                cmax=max(temperatures_sorted),
                colorbar=dict(
                    title=dict(
                        text="<b>Temperature<br>[K]</b>",
                        font=dict(
                            size=COLORBAR_TITLE_SIZE,
                            family=FONT_FAMILY,
                            color="black"
                        )
                    ),
                    tickfont=dict(
                        size=COLORBAR_TICK_SIZE,
                        family=FONT_FAMILY,
                        color="black"
                    ),
                    x=1.06,
                    y=0.72,
                    len=0.56,
                    thickness=28,
                    outlinewidth=2.8,
                    outlinecolor="black",
                    borderwidth=2.2,
                    bordercolor="black",
                    bgcolor="white",
                    tickmode="array",
                    tickvals=temp_tickvals,
                    ticktext=temp_ticktext,
                    tickformat=".0f"
                ),
                line=dict(
                    color=RING_BORDER_COLOR,
                    width=RING_BORDER_WIDTH
                )
            ),
            customdata=temp_custom,
            hovertemplate=(
                "<b>TEMPERATURE RING</b><br><br>"
                "<b>T</b> = %{customdata[0]} K<br>"
                "<b>Normalized T</b> = %{customdata[1]:.3f}"
                "<extra></extra>"
            ),
            name="<b>Temperature</b>",
            showlegend=False,
            opacity=0.98
        )
    )

    # ============================================================
    # MIDDLE COMPOSITION RING
    # ============================================================

    for el in ["Co", "Cr", "Fe", "Ni"]:

        fig.add_trace(
            go.Barpolar(
                theta=element_data[el]["theta"],
                width=element_data[el]["width"],
                r=element_data[el]["r"],
                base=element_data[el]["base"],
                marker=dict(
                    color=comp_colors_map[el],
                    line=dict(
                        color=COMPOSITION_BORDER_COLOR,
                        width=COMPOSITION_BORDER_WIDTH
                    )
                ),
                customdata=element_data[el]["custom"],
                hovertemplate=(
                    f"<b>{el} COMPOSITION LAYER</b><br><br>"
                    "T = %{customdata[0]} K<br>"
                    "<b>Current element</b> = %{customdata[8]}<br>"
                    "<b>Current element fraction</b> = %{customdata[9]:.3f}<br><br>"
                    "<b>Co</b> = %{customdata[1]:.3f}<br>"
                    "<b>Cr</b> = %{customdata[2]:.3f}<br>"
                    "<b>Fe</b> = %{customdata[3]:.3f}<br>"
                    "<b>Ni</b> = %{customdata[4]:.3f}<br><br>"
                    "ΔG = %{customdata[5]:.2f} J/mol<br>"
                    "P_chem = %{customdata[6]:.3f} MPa<br>"
                    "P_net = %{customdata[10]:.3f} MPa<br>"
                    "Mode = %{customdata[11]}<br>"
                    "<b>Differential Force</b> = %{customdata[7]:.2e} N"
                    "<extra></extra>"
                ),
                name=f"<b>{el}</b>",
                showlegend=True,
                opacity=0.98
            )
        )

    # ============================================================
    # OUTER FORCE RING
    # ============================================================

    fig.add_trace(
        go.Barpolar(
            theta=force_theta,
            width=force_widths,
            r=force_r,
            base=force_base,
            marker=dict(
                color=force_colors,
                colorscale=force_cmap,
                cmin=-max_abs_force,
                cmax=max_abs_force,
                cmid=0,
                colorbar=dict(
                    title=dict(
                        text="<b>Differential Force<br>[N]</b>",
                        font=dict(
                            size=COLORBAR_TITLE_SIZE,
                            family=FONT_FAMILY,
                            color="black"
                        )
                    ),
                    tickfont=dict(
                        size=COLORBAR_TICK_SIZE,
                        family=FONT_FAMILY,
                        color="black"
                    ),
                    x=1.23,
                    y=0.72,
                    len=0.56,
                    thickness=28,
                    outlinewidth=2.8,
                    outlinecolor="black",
                    borderwidth=2.2,
                    bordercolor="black",
                    bgcolor="white",
                    tickmode="array",
                    tickvals=force_tickvals,
                    ticktext=force_ticktext,
                    tickformat=".1e"
                ),
                line=dict(
                    color=RING_BORDER_COLOR,
                    width=RING_BORDER_WIDTH
                )
            ),
            customdata=force_custom,
            hovertemplate=(
                "<b>FORCE RING</b><br><br>"
                "T = %{customdata[0]} K<br>"
                "<b>Co</b> = %{customdata[1]:.3f}<br>"
                "<b>Cr</b> = %{customdata[2]:.3f}<br>"
                "<b>Fe</b> = %{customdata[3]:.3f}<br>"
                "<b>Ni</b> = %{customdata[4]:.3f}<br><br>"
                "ΔG = %{customdata[5]:.2f} J/mol<br>"
                "P_chem = %{customdata[6]:.3f} MPa<br>"
                "P_net = %{customdata[8]:.3f} MPa<br>"
                "Mode = %{customdata[9]}<br>"
                "<b>Differential Force</b> = %{customdata[7]:.2e} N"
                "<extra></extra>"
            ),
            name="<b>dF_net</b>",
            showlegend=False,
            opacity=0.98
        )
    )

    # ============================================================
    # ANNOTATIONS
    # ============================================================

    fig.add_annotation(
        text="<b>INNER RING: TEMPERATURE</b>",
        x=0.02,
        y=1.065,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=17,
            family=FONT_FAMILY,
            color="black"
        ),
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="black",
        borderwidth=1.6,
        borderpad=5
    )

    fig.add_annotation(
        text="<b>MIDDLE RING: Co / Cr / Fe / Ni COMPOSITION</b>",
        x=0.39,
        y=1.065,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=17,
            family=FONT_FAMILY,
            color="black"
        ),
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="black",
        borderwidth=1.6,
        borderpad=5
    )

    fig.add_annotation(
        text="<b>OUTER RING: dF_net [N]</b>",
        x=0.79,
        y=1.065,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=17,
            family=FONT_FAMILY,
            color="black"
        ),
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="black",
        borderwidth=1.6,
        borderpad=5
    )

    # ============================================================
    # LAYOUT
    # ============================================================

    fig.update_layout(
        title=dict(
            text=(
                "<b>Temperature → Composition → Interface Driving Force</b><br>"
                f"<span style='font-size:{SUBTITLE_FONT_SIZE}px'>"
                "Inner ring: Temperature [K] | "
                "Middle ring: Co–Cr–Fe–Ni mole fractions | "
                f"Outer ring: Differential Force dF_net [N] | dV = {dV:.2e} m³"
                "</span>"
            ),
            x=0.5,
            y=0.975,
            xanchor="center",
            yanchor="top",
            font=dict(
                size=TITLE_FONT_SIZE,
                family=FONT_FAMILY,
                color="black"
            )
        ),
        template="plotly_white",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        height=1080,
        width=1250,
        margin=dict(
            t=160,
            b=45,
            l=45,
            r=285
        ),
        font=dict(
            family=FONT_FAMILY,
            size=18,
            color="black"
        ),
        legend=dict(
            title=dict(
                text="<b>Composition</b>",
                font=dict(
                    size=22,
                    family=FONT_FAMILY,
                    color="black"
                )
            ),
            font=dict(
                size=LEGEND_FONT_SIZE,
                family=FONT_FAMILY,
                color="black"
            ),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=2.2,
            x=1.05,
            y=0.24,
            xanchor="left",
            yanchor="middle",
            itemsizing="constant"
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font=dict(
                size=HOVER_FONT_SIZE,
                family=FONT_FAMILY,
                color="black"
            )
        ),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                visible=False,
                range=[0, force_ring_base + force_ring_thickness + 0.30],
                showline=False,
                showgrid=False
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                showticklabels=False,
                ticks="",
                showline=False,
                gridcolor="rgba(120,120,120,0.22)",
                gridwidth=1.0
            )
        )
    )

    fig.update_polars(sector=[0, 360])

    st.plotly_chart(fig, width='stretch')

    # ============================================================
    # TRANSPARENT HIGH-RES PNG DOWNLOAD
    # ============================================================

    try:
        export_fig = go.Figure(fig)

        export_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        export_fig.update_polars(
            bgcolor="rgba(0,0,0,0)"
        )

        png_bytes = export_fig.to_image(
            format="png",
            width=2400,
            height=2200,
            scale=1
        )

        st.download_button(
            label="📥 Download Transparent High-Resolution PNG",
            data=png_bytes,
            file_name="CoCrFeNi_temperature_composition_force_multiring_transparent.png",
            mime="image/png"
        )

    except Exception as exc:
        st.warning(
            "PNG export failed. Install Kaleido if you need export: `pip install -U kaleido`. "
            "The interactive chart above still works."
        )
        with st.expander("Show export error"):
            st.code(str(exc))


# ============================================================
# DEBUG SUMMARY
# ============================================================

with st.expander("Debug summary", expanded=False):
    st.write({
        "csv_folder": CSV_FILES_DIR,
        "temperature_order": temperatures_sorted,
        "n_temperatures": n_temp,
        "temperature_sector_width_deg": temp_width,
        "n_force_segments": len(force_theta),
        "differential_force_min_N": float(np.min(force_colors)) if len(force_colors) > 0 else None,
        "differential_force_max_N": float(np.max(force_colors)) if len(force_colors) > 0 else None,
        "max_abs_differential_force_N": float(max_abs_force),
        "composition_colors": comp_colors_map,
        "ring_geometry": {
            "temperature": {
                "base": temp_ring_base,
                "thickness": temp_ring_thickness
            },
            "composition": {
                "base": comp_ring_base,
                "thickness": comp_ring_thickness
            },
            "force": {
                "base": force_ring_base,
                "thickness": force_ring_thickness
            }
        }
    })
