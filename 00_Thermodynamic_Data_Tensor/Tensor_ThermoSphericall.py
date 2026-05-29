import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
import uuid
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, Delaunay

# Try importing scipy.special
try:
    import scipy.special as special
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ `scipy` not available. SH and advanced modes disabled.")

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILES_DIR = os.path.join(SCRIPT_DIR, "csv_files")
os.makedirs(CSV_FILES_DIR, exist_ok=True)

st.set_page_config(page_title="CoCrFeNi Phase Stability Explorer v2", layout="wide")

# =============================================
# COLOR & SYMBOL LIBRARY
# =============================================
COLORMAPS = sorted(list(set([
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo",
    "Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys", "Oranges", "OrRd",
    "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu", "Reds", "YlGn", "YlGnBu",
    "YlOrBr", "YlOrRd", "BrBG", "PRGn", "PiYG", "PuOr", "RdBu", "RdGy",
    "RdYlBu", "RdYlGn", "Spectral", "Twilight", "HSV", "Jet", "Rainbow",
    "Hot", "Cool", "Blackbody", "Electric", "Plotly3", "Portland", "Picnic",
    "Solar", "Balance", "Delta", "Curl", "IceFire", "Edge", "Fall", "Sunset",
    "Sunsetdark", "Teal", "Tealgrn", "Tropic", "Peach", "Oxy", "Mint",
    "Emrld", "Aggrnyl", "Agsunset", "Armyrose", "Bluered", "Blugrn", "Bluyl",
    "Brwnyl", "Burg", "Burgyl", "Darkmint", "Geysr", "Magenta", "Mrybm",
    "Mygbm", "Oryel", "Pinkyl", "Purp", "Purpor", "Redor", "Ylorrd", "Ylorbr",
    "Ylgnbu", "Ylgn", "Haline", "Ice", "Matter", "Speed", "Tempo", "Thermal",
    "Turbid", "Algae", "Deep", "Dense", "Sinebow", "Phase"
])))

PHASE_SYMBOLS = {"LIQUID": "circle", "FCC": "diamond", "BOUNDARY": "x"}
PHASE_COLORS = {"LIQUID": "#e74c3c", "FCC": "#2980b9", "BOUNDARY": "#f1c40f"}
PHASE_COLORS_RGBA = {"LIQUID": "rgba(231, 76, 60, 0.25)", "FCC": "rgba(41, 128, 185, 0.25)"}


# =============================================
# VISUAL STYLE HELPERS
# =============================================
def radial_origin_offset(distance):
    """Move plotted SH/surface objects away from the coordinate origin along the Co-Cr-Fe diagonal."""
    d = float(distance) / np.sqrt(3.0)
    return d, d, d

def shifted_xyz(X, Y, Z, distance):
    """Apply the radial origin offset to a plotted surface/line."""
    ox, oy, oz = radial_origin_offset(distance)
    return X + ox, Y + oy, Z + oz

def make_surface_customdata(Co, Cr, Fe, Ni, G_liq, G_fcc, dG):
    """Stack thermodynamic/composition fields for richer Plotly hover labels."""
    return np.stack([Co, Cr, Fe, Ni, G_liq, G_fcc, dG], axis=-1)

def thermo_hovertemplate(phase_label, value_label="G"):
    return (
        f"<b>{phase_label}</b><br>"
        "Co=%{customdata[0]:.3f}<br>"
        "Cr=%{customdata[1]:.3f}<br>"
        "Fe=%{customdata[2]:.3f}<br>"
        "Ni=%{customdata[3]:.3f}<br>"
        "G_LIQ=%{customdata[4]:,.0f} J/mol<br>"
        "G_FCC=%{customdata[5]:,.0f} J/mol<br>"
        "ΔG=G_LIQ-G_FCC=%{customdata[6]:,.0f} J/mol<br>"
        f"{value_label}=%{{surfacecolor:,.0f}} J/mol<br>"
        f"T={T_val} K"
        "<extra></extra>"
    )

# =============================================
# DATA LOADING
# =============================================
@st.cache_data
def load_all_data(csv_dir=CSV_FILES_DIR):
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*.csv")))
    if not files:
        st.error(f"No CSV files found in `{csv_dir}`.")
        st.stop()
    dfs = []
    for f in files:
        basename = os.path.basename(f)
        try:
            T = int(basename.replace("Gibbs_", "").replace("K.csv", ""))
            df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
            df["T"] = T
            dfs.append(df)
        except Exception as e:
            st.warning(f"Skipping {f}: {e}")
    return pd.concat(dfs, ignore_index=True)

df = load_all_data()

T_list = sorted(df["T"].unique())
T_min = min(T_list)
T_max = max(T_list)
T_range = T_max - T_min if T_max > T_min else 1.0

# Global ranges for consistent scaling
G_LIQ_global_min = df["G_LIQ"].min()
G_LIQ_global_max = df["G_LIQ"].max()
G_FCC_global_min = df["G_FCC"].min()
G_FCC_global_max = df["G_FCC"].max()
G_global_min = min(G_LIQ_global_min, G_FCC_global_min)
G_global_max = max(G_LIQ_global_max, G_FCC_global_max)

df["dG"] = df["G_LIQ"] - df["G_FCC"]
dG_global_min = df["dG"].min()
dG_global_max = df["dG"].max()
dG_global_abs_max = max(abs(dG_global_min), abs(dG_global_max))

# Build convex hull of all data points for uncertainty/distance calculation
all_pts = df[["Co", "Cr", "Fe"]].values
try:
    data_hull = ConvexHull(all_pts)
    HULL_AVAILABLE = True
except Exception:
    HULL_AVAILABLE = False
    data_hull = None

# =============================================
# INTERPOLATION
# =============================================
@st.cache_data(ttl=3600)
def build_interpolators_for_T(df, T):
    df_T = df[df["T"] == T].copy()
    if len(df_T) == 0:
        return None, None
    pts = df_T[["Co", "Cr", "Fe"]].values
    interp_liq = LinearNDInterpolator(pts, df_T["G_LIQ"].values)
    interp_fcc = LinearNDInterpolator(pts, df_T["G_FCC"].values)
    return interp_liq, interp_fcc

# =============================================
# TETRAHEDRAL GRID & UNCERTAINTY
# =============================================
def generate_tetrahedral_grid(resolution=25):
    x = np.linspace(0, 1, resolution)
    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
    grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
    valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
    return grid_pts[valid_mask]

def compute_data_proximity(pts, data_pts, max_dist=0.15):
    """Compute normalized proximity to nearest data point (1 = on data, 0 = far)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(data_pts)
    dists, _ = tree.query(pts, k=1)
    proximity = np.clip(1.0 - dists / max_dist, 0.0, 1.0)
    return proximity

def find_phase_boundary_points(pts, dG_values, threshold=50.0):
    boundary_mask = np.abs(dG_values) < threshold
    return pts[boundary_mask], dG_values[boundary_mask]

# =============================================
# SPHERICAL HARMONICS (Enhanced)
# =============================================
if SCIPY_AVAILABLE:
    def get_real_sph_harm(l, m, theta, phi):
        if hasattr(special, 'sph_harm_y'):
            Y_complex = special.sph_harm_y(l, m, phi, theta)
        else:
            Y_complex = special.sph_harm(m, l, theta, phi)
        if m > 0:
            return np.sqrt(2.0) * Y_complex.real
        elif m < 0:
            if hasattr(special, 'sph_harm_y'):
                Y_pos = special.sph_harm_y(l, abs(m), phi, theta)
            else:
                Y_pos = special.sph_harm(abs(m), l, theta, phi)
            return np.sqrt(2.0) * Y_pos.imag
        else:
            return Y_complex.real

    def sample_g_on_sphere(interp_liq, interp_fcc, R_fixed, n_theta=60, n_phi=60):
        theta = np.linspace(0, 2*np.pi, n_theta)
        phi = np.linspace(0, np.pi, n_phi)
        TH, PH = np.meshgrid(theta, phi)
        x = R_fixed * np.sin(PH) * np.cos(TH)
        y = R_fixed * np.sin(PH) * np.sin(TH)
        z = R_fixed * np.cos(PH)
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        valid = (pts[:,0] + pts[:,1] + pts[:,2]) <= 1.0
        G_liq = interp_liq(pts) if interp_liq is not None else np.full(len(pts), np.nan)
        G_fcc = interp_fcc(pts) if interp_fcc is not None else np.full(len(pts), np.nan)
        G_stable = np.where(G_liq <= G_fcc, G_liq, G_fcc)
        dG = G_liq - G_fcc
        valid = valid & ~np.isnan(G_stable)
        return TH, PH, G_stable.reshape(TH.shape), dG.reshape(TH.shape), valid.reshape(TH.shape), pts

    @st.cache_data(ttl=3600)
    def fit_sh_coeffs(theta_vals, phi_vals, g_vals, l_max=3):
        theta_flat = theta_vals.ravel()
        phi_flat = phi_vals.ravel()
        g_flat = g_vals.ravel()
        valid = ~np.isnan(g_flat)
        theta_flat = theta_flat[valid]
        phi_flat = phi_flat[valid]
        g_flat = g_flat[valid]
        if len(theta_flat) == 0:
            return None, l_max
        A = []
        for t, p in zip(theta_flat, phi_flat):
            row = []
            for l in range(l_max+1):
                for m in range(-l, l+1):
                    y = get_real_sph_harm(l, m, t, p)
                    row.append(y)
            A.append(row)
        A = np.array(A)
        from scipy.linalg import lstsq
        coeffs, _, _, _ = lstsq(A, g_flat)
        return coeffs, l_max

    def reconstruct_sh_surface(theta_grid, phi_grid, coeffs, l_max):
        recon = np.zeros_like(theta_grid, dtype=float)
        idx = 0
        for l in range(l_max+1):
            for m in range(-l, l+1):
                Y = get_real_sph_harm(l, m, theta_grid, phi_grid)
                recon += coeffs[idx] * Y
                idx += 1
        return recon

    def extract_dg_zero_contour(TH, PH, dG_grid, R_fixed):
        """Robust ΔG=0 contour extraction with edge-walking and interpolation."""
        contours_x, contours_y, contours_z = [], [], []
        # Horizontal edges
        for i in range(dG_grid.shape[0]):
            for j in range(dG_grid.shape[1]-1):
                if not (np.isfinite(dG_grid[i,j]) and np.isfinite(dG_grid[i,j+1])):
                    continue
                if dG_grid[i,j] * dG_grid[i,j+1] < 0:
                    t = abs(dG_grid[i,j]) / (abs(dG_grid[i,j]) + abs(dG_grid[i,j+1]) + 1e-12)
                    th_mid = TH[i,j] + t * (TH[i,j+1] - TH[i,j])
                    ph_mid = PH[i,j] + t * (PH[i,j+1] - PH[i,j])
                    r = R_fixed
                    contours_x.append(r * np.sin(ph_mid) * np.cos(th_mid))
                    contours_y.append(r * np.sin(ph_mid) * np.sin(th_mid))
                    contours_z.append(r * np.cos(ph_mid))
        # Vertical edges
        for i in range(dG_grid.shape[0]-1):
            for j in range(dG_grid.shape[1]):
                if not (np.isfinite(dG_grid[i,j]) and np.isfinite(dG_grid[i+1,j])):
                    continue
                if dG_grid[i,j] * dG_grid[i+1,j] < 0:
                    t = abs(dG_grid[i,j]) / (abs(dG_grid[i,j]) + abs(dG_grid[i+1,j]) + 1e-12)
                    th_mid = TH[i,j] + t * (TH[i+1,j] - TH[i,j])
                    ph_mid = PH[i,j] + t * (PH[i+1,j] - PH[i,j])
                    r = R_fixed
                    contours_x.append(r * np.sin(ph_mid) * np.cos(th_mid))
                    contours_y.append(r * np.sin(ph_mid) * np.sin(th_mid))
                    contours_z.append(r * np.cos(ph_mid))
        return np.array(contours_x), np.array(contours_y), np.array(contours_z)

# =============================================
# TEMPERATURE-DRIVEN DEFORMATION FUNCTIONS
# =============================================
def get_liquid_radius(G_sh, sh_R_fixed, T_factor):
    """Fluid, expanded, smooth — stronger at high T."""
    g_min, g_max = np.nanmin(G_sh), np.nanmax(G_sh)
    norm = (G_sh - g_min) / (g_max - g_min + 1e-12) if g_max > g_min else np.zeros_like(G_sh)
    thermal_exp = 1.0 + 0.35 * T_factor
    fluid_dist = 0.12 * np.sin(2 * np.pi * norm) * (0.5 + 0.5 * T_factor)
    return sh_R_fixed * (thermal_exp + 0.22 * norm + fluid_dist)

def get_fcc_radius(G_sh, sh_R_fixed, T_factor):
    """Crystalline, faceted, rigid — stronger at low T."""
    g_min, g_max = np.nanmin(G_sh), np.nanmax(G_sh)
    norm = (G_sh - g_min) / (g_max - g_min + 1e-12) if g_max > g_min else np.zeros_like(G_sh)
    rigidity = 1.0 - 0.20 * T_factor
    crystal_factor = 0.28 * (1.0 - T_factor)
    # Multiple harmonics for crystalline feel
    crystal_ripples = crystal_factor * (
        0.6 * np.sin(6 * np.pi * norm) +
        0.3 * np.sin(10 * np.pi * norm) +
        0.1 * np.sin(14 * np.pi * norm)
    )
    return sh_R_fixed * (rigidity + 0.20 * norm + crystal_ripples)

# =============================================
# HEADER
# =============================================
st.title("🔷 Co-Cr-Fe-Ni Phase Stability Explorer v2")
st.markdown(r"""
**Single-Temperature Phase Comparison with Temperature-Driven Shape Morphing.**  
FCC surfaces become **crystalline & faceted** at low T. LIQUID surfaces become **fluid & expanded** at high T.  
The **ΔG = 0 boundary** (gold) marks the exact transition frontier.
""")

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.header("🎛️ Control Panel")

    # --- PRESET VIEWS ---
    st.subheader("⚡ Quick Presets")
    preset = st.selectbox("Load Preset", [
        "Custom", "Low-T FCC Crystal", "High-T Liquid Melt", 
        "Transition Region", "Maximum Contrast"
    ], index=0)

    # --- TEMPERATURE ---
    st.subheader("🌡️ Temperature")
    if preset == "Low-T FCC Crystal":
        default_T = T_min
    elif preset == "High-T Liquid Melt":
        default_T = T_max
    elif preset == "Transition Region":
        default_T = T_list[len(T_list)//2] if T_list else 1000
    else:
        default_T = T_list[len(T_list)//2] if T_list else 1000

    T_val = st.select_slider("T (K)", options=T_list, value=default_T)
    T_factor = (T_val - T_min) / T_range if T_range > 0 else 0.5
    phase_expected = "LIQUID" if T_factor > 0.6 else "FCC" if T_factor < 0.4 else "Transition"
    st.info(f"T = {T_val}K | Expected: **{phase_expected}**")

    st.divider()

    # --- QUERY ---
    st.subheader("📍 Query Composition")
    q_co = st.number_input("x_Co", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    q_cr = st.number_input("x_Cr", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    q_fe = st.number_input("x_Fe", 0.0, 1.0, 0.25, 0.01, format="%.2f")
    comp_sum = q_co + q_cr + q_fe
    if comp_sum > 1.0:
        st.warning(f"⚠️ Sum = {comp_sum:.2f} > 1.0")
    eval_query = st.button("🔍 Evaluate", width='stretch')

    st.divider()

    # --- VISUALIZATION MODE ---
    st.subheader("🎨 Visualization Mode")
    mode_options = [
        "Phase Boundary (Scientific)",
        "Dual SH Surfaces (Temperature Morph)",
        "ΔG Difference Surface",
        "Ternary Flat Projection",
        "Markers (Distinct Shapes)",
        "Animated Temperature Sweep"
    ]
    if not SCIPY_AVAILABLE:
        mode_options = [m for m in mode_options if "SH" not in m and "Difference" not in m and "Animated" not in m]
        st.error("SciPy missing: Advanced modes disabled")

    render_mode = st.radio("Mode", mode_options, index=1 if SCIPY_AVAILABLE else 0)

    st.divider()

    # --- MODE CONTROLS ---
    if render_mode == "Phase Boundary (Scientific)":
        st.subheader("🔧 Scientific Settings")
        grid_res = st.slider("Grid Resolution", 15, 80, 35, step=5)
        boundary_threshold = st.slider("Boundary Width (J/mol)", 10, 300, 60, 10)
        show_phase_volume = st.toggle("Show Phase Volume", value=True)
        volume_opacity = st.slider("Volume Opacity", 0.05, 0.6, 0.12, 0.05)
        volume_size = st.slider("Volume Point Size", 1, 8, 2)
        show_uncertainty = st.toggle("Fade Uncertain Regions", value=True)
        uncertainty_fade = st.slider("Fade Strength", 0.0, 1.0, 0.6, 0.1)
        show_simplex = st.toggle("Show Simplex Frame", value=True)
        show_slice = st.toggle("Show Cross-Section Plane", value=False)
        slice_ni = st.slider("Slice x_Ni", 0.0, 1.0, 0.25, 0.05) if show_slice else 0.25

    elif render_mode == "Dual SH Surfaces (Temperature Morph)":
        st.subheader("🔧 SH Morph Settings")

        if preset == "Low-T FCC Crystal":
            sh_R_fixed = 0.45
            sh_l_max = 5
            liq_opacity = 0.35
            fcc_opacity = 0.85
        elif preset == "High-T Liquid Melt":
            sh_R_fixed = 0.65
            sh_l_max = 2
            liq_opacity = 0.85
            fcc_opacity = 0.25
        elif preset == "Maximum Contrast":
            sh_R_fixed = 0.50
            sh_l_max = 4
            liq_opacity = 0.70
            fcc_opacity = 0.70
        else:
            sh_R_fixed = 0.50
            sh_l_max = 3
            liq_opacity = 0.60
            fcc_opacity = 0.45

        sh_R_fixed = st.slider("Base Radius", 0.1, 0.9, sh_R_fixed, 0.01)

        # Per-phase l_max with T-dependence
        l_max_liq = max(1, int(sh_l_max - 1.5 * T_factor))
        l_max_fcc = max(2, int(sh_l_max + 1.0 * (1.0 - T_factor)))

        st.markdown(f"**Auto l_max:** LIQUID l={l_max_liq} (smooth), FCC l={l_max_fcc} (faceted)")
        sh_l_max_override = st.slider("Override l_max (base)", 1, 8, sh_l_max)
        if sh_l_max_override != sh_l_max:
            l_max_liq = max(1, int(sh_l_max_override - 1.5 * T_factor))
            l_max_fcc = max(2, int(sh_l_max_override + 1.0 * (1.0 - T_factor)))

        sh_n_theta = st.slider("Theta Resolution", 30, 150, 70, step=10)
        sh_n_phi = st.slider("Phi Resolution", 30, 150, 70, step=10)

        liq_cmap = st.selectbox(
            "LIQUID Colormap", COLORMAPS,
            index=COLORMAPS.index("Reds") if "Reds" in COLORMAPS else 0
        )
        fcc_cmap = st.selectbox(
            "FCC Colormap", COLORMAPS,
            index=COLORMAPS.index("Blues") if "Blues" in COLORMAPS else 0
        )

        liq_opacity = st.slider("LIQUID Opacity", 0.0, 1.0, liq_opacity, 0.05)
        fcc_opacity = st.slider("FCC Opacity", 0.0, 1.0, fcc_opacity, 0.05)
        surface_origin_radius = st.slider(
            "Surface Origin Radial Distance",
            0.0, 1.5, 0.0, 0.05,
            help="Moves the SH surfaces away from the coordinate origin along the positive Co-Cr-Fe diagonal."
        )
        show_dg_contour = st.toggle("Show ΔG=0 Contour", value=True)
        show_data_density = st.toggle("Show Data Coverage", value=False)

        st.markdown("""
        <small>
        <b>Low T:</b> FCC = rigid, crystalline ripples; LIQUID = small, faint.<br>
        <b>High T:</b> LIQUID = expanded, fluid, shiny; FCC = shrunk, matte.
        </small>
        """, unsafe_allow_html=True)

    elif render_mode == "ΔG Difference Surface":
        st.subheader("🔧 ΔG Surface Settings")
        sh_R_fixed = st.slider("Base Radius", 0.2, 0.9, 0.50, 0.05)
        sh_l_max = st.slider("Max Harmonic Degree", 1, 8, 4)
        sh_n_theta = st.slider("Theta Resolution", 30, 150, 70, step=10)
        sh_n_phi = st.slider("Phi Resolution", 30, 150, 70, step=10)
        dg_cmap = st.selectbox(
            "ΔG Surface Colormap", COLORMAPS,
            index=COLORMAPS.index("RdBu") if "RdBu" in COLORMAPS else 0
        )
        dg_opacity = st.slider("ΔG Surface Opacity", 0.0, 1.0, 0.90, 0.05)
        surface_origin_radius = st.slider(
            "Surface Origin Radial Distance",
            0.0, 1.5, 0.0, 0.05,
            help="Moves the ΔG surface away from the coordinate origin along the positive Co-Cr-Fe diagonal."
        )
        dg_scale = st.slider("ΔG Deformation Scale", 0.001, 0.15, 0.025, 0.001)
        show_dg_contour = st.toggle("Show ΔG=0 Contour", value=True)

    elif render_mode == "Ternary Flat Projection":
        st.subheader("🔧 Ternary Settings")
        flat_color_by = st.radio("Color By", ["Stable Phase", "ΔG (diverging)", "G_magnitude", "Data Proximity"], index=1)
        flat_marker_size = st.slider("Marker Size", 2, 20, 7)
        flat_opacity = st.slider("Opacity", 0.1, 1.0, 0.85, 0.05)
        show_ternary_grid = st.toggle("Grid Lines", value=True)
        show_uncertainty = st.toggle("Fade Distant Points", value=True)

    elif render_mode == "Markers (Distinct Shapes)":
        st.subheader("🔧 Marker Settings")
        grid_res = st.slider("Grid Resolution", 15, 100, 35, step=5)
        marker_size = st.slider("Marker Size", 1, 12, 4)
        opacity = st.slider("Opacity", 0.1, 1.0, 0.85, 0.05)
        show_phase = st.radio("Display", ["Stable Phase Only", "Both Phases (Distinct)"], index=1)
        cmap = st.selectbox("Colormap", COLORMAPS, index=COLORMAPS.index("RdBu_r") if "RdBu_r" in COLORMAPS else 0)
        show_boundary = st.toggle("Show ΔG≈0 Boundary", value=True)
        show_uncertainty = st.toggle("Fade Distant Points", value=True)

    else:  # Animated Temperature Sweep
        st.subheader("🔧 Animation Settings")
        anim_start = st.select_slider("Start T", options=T_list, value=T_min)
        anim_end = st.select_slider("End T", options=T_list, value=T_max)
        anim_frames = st.slider("Frames", 3, min(20, len(T_list)), min(8, len(T_list)))
        anim_mode = st.radio("Animation Style", ["Dual SH Morph", "ΔG Surface Morph"], index=0)
        sh_R_fixed = st.slider("Base Radius", 0.2, 0.9, 0.50, 0.05)
        sh_l_max = st.slider("l_max", 1, 6, 3)
        sh_n_theta = st.slider("Resolution", 30, 100, 50, step=10)
        liq_cmap = st.selectbox(
            "LIQUID Colormap", COLORMAPS,
            index=COLORMAPS.index("Reds") if "Reds" in COLORMAPS else 0,
            key="anim_liq_cmap"
        )
        fcc_cmap = st.selectbox(
            "FCC Colormap", COLORMAPS,
            index=COLORMAPS.index("Blues") if "Blues" in COLORMAPS else 0,
            key="anim_fcc_cmap"
        )
        surface_origin_radius = st.slider(
            "Surface Origin Radial Distance", 0.0, 1.5, 0.0, 0.05,
            help="Moves the animated surfaces away from the coordinate origin along the positive Co-Cr-Fe diagonal."
        )

    st.divider()

    # --- GLOBAL EXTRAS ---
    st.subheader("🔷 Overlays")
    show_axes_frame = st.toggle("Coordinate Axes", value=True)
    axes_origin_radius = st.slider(
        "Coordinate Axes Origin Radial Distance",
        0.0, 1.5, 0.0, 0.05,
        help="Moves only the red/green/blue Co-Cr-Fe coordinate axes away from the true origin along the positive Co-Cr-Fe diagonal. This is independent of Surface Origin Radial Distance."
    )
    show_query_probe = st.toggle("Query Probe Sphere", value=True)
    show_comp_path = st.toggle("Show Composition Path", value=False,
                                help="If you query multiple points, traces a line between them")

    st.divider()
    st.subheader("✏️ Layout")
    template = st.selectbox("Template", ["plotly_white", "plotly_dark", "seaborn", "simple_white"], index=0)
    bg_color = st.color_picker("Background", "#ffffff")
    title_font = st.slider("Title Font", 16, 42, 28)
    axis_tick_font = st.slider("Axis Tick Font", 10, 28, 18)
    axis_tick_length = st.slider(
        "Axis Tick Padding / Length", 0, 30, 14, 1,
        help="Moves tick labels farther from the 3D axis line by increasing the outward tick length."
    )
    coord_axis_label_font = st.slider("Coordinate Axes Label Font", 14, 42, 28)
    coord_axis_line_width = st.slider("Coordinate Axes Line Width", 3, 16, 9)
    colorbar_title_font = st.slider("Colorbar Title Font", 12, 32, 22)
    colorbar_tick_font = st.slider("Colorbar Tick Font", 10, 28, 18)
    plot_width = st.slider("Figure Width (px)", 900, 2400, 1500, 50)
    plot_height = st.slider("Figure Height (px)", 650, 1800, 1050, 50)

    st.divider()
    st.caption(f"Data: {len(T_list)} temperatures | {len(df):,} rows")

# =============================================
# SESSION STATE FOR COMPOSITION PATH
# =============================================
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# =============================================
# QUERY EVALUATION
# =============================================
query_result = None
if eval_query:
    if comp_sum > 1.0:
        st.error("Composition sum exceeds 1.0")
    else:
        interp_liq_q, interp_fcc_q = build_interpolators_for_T(df, T_val)
        if interp_liq_q is not None:
            pt = np.array([[q_co, q_cr, q_fe]])
            g_liq_q = float(interp_liq_q(pt)[0])
            g_fcc_q = float(interp_fcc_q(pt)[0])
            if np.isnan(g_liq_q) or np.isnan(g_fcc_q):
                st.error("Query point outside data convex hull")
            else:
                g_stable_q = min(g_liq_q, g_fcc_q)
                phase_q = "LIQUID" if g_liq_q <= g_fcc_q else "FCC"
                dG_q = g_liq_q - g_fcc_q
                query_result = {
                    "T": T_val, "Co": q_co, "Cr": q_cr, "Fe": q_fe,
                    "Ni": 1.0 - comp_sum,
                    "G_LIQ": g_liq_q, "G_FCC": g_fcc_q,
                    "G_stable": g_stable_q, "Phase": phase_q, "dG": dG_q
                }
                st.session_state.query_history.append(query_result)
                if len(st.session_state.query_history) > 10:
                    st.session_state.query_history.pop(0)
        else:
            st.error(f"No data for T={T_val}K")

if query_result:
    st.success(f"Query at T={query_result['T']}K, x_Ni={query_result['Ni']:.3f}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("G_LIQ", f"{query_result['G_LIQ']:,.0f}", "J/mol")
    c2.metric("G_FCC", f"{query_result['G_FCC']:,.0f}", "J/mol")
    c3.metric("ΔG", f"{query_result['dG']:,.0f}", "J/mol",
              delta_color="inverse" if query_result['dG'] < 0 else "normal")
    c4.metric("Stable", query_result['Phase'])
    c5.metric("|ΔG|", f"{abs(query_result['dG']):,.0f}", "J/mol")
    st.divider()

# =============================================
# MAIN RENDERING
# =============================================
interp_liq, interp_fcc = build_interpolators_for_T(df, T_val)
if interp_liq is None:
    st.error(f"No interpolator for T={T_val}K")
    st.stop()

fig = go.Figure()

# ------------------------------------------------------------------
# MODE 1: PHASE BOUNDARY (SCIENTIFIC)
# ------------------------------------------------------------------
if render_mode == "Phase Boundary (Scientific)":
    pts = generate_tetrahedral_grid(grid_res)
    G_liq = interp_liq(pts)
    G_fcc = interp_fcc(pts)
    valid = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
    pts = pts[valid]
    G_liq = G_liq[valid]
    G_fcc = G_fcc[valid]
    dG = G_liq - G_fcc
    stable = np.where(dG <= 0, "LIQUID", "FCC")

    # Uncertainty / data proximity
    if show_uncertainty and HULL_AVAILABLE:
        proximity = compute_data_proximity(pts, all_pts, max_dist=0.2)
    else:
        proximity = np.ones(len(pts))

    # Phase volume with distinct shapes
    for phase in ["LIQUID", "FCC"]:
        mask = stable == phase
        if mask.sum() == 0:
            continue
        p_pts = pts[mask]
        p_prox = proximity[mask]
        p_dG = dG[mask]

        fig.add_trace(go.Scatter3d(
            x=p_pts[:, 0], y=p_pts[:, 1], z=p_pts[:, 2],
            mode="markers",
            marker=dict(
                size=volume_size,
                color=PHASE_COLORS[phase],
                symbol=PHASE_SYMBOLS[phase],
                opacity=volume_opacity * p_prox,
                line=dict(width=0.5, color="white")
            ),
            name=f"{phase} Region",
            hovertemplate=(f"<b>{phase}</b><br>" +
                           "x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Fe=%{z:.3f}<br>" +
                           "ΔG=%{customdata:.0f} J/mol<extra></extra>"),
            customdata=p_dG
        ))

    # Boundary
    boundary_pts, boundary_dG = find_phase_boundary_points(pts, dG, boundary_threshold)
    if len(boundary_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=boundary_pts[:, 0], y=boundary_pts[:, 1], z=boundary_pts[:, 2],
            mode="markers",
            marker=dict(size=5, color=PHASE_COLORS["BOUNDARY"], symbol="x",
                        line=dict(width=2, color="#b7950b")),
            name="ΔG = 0 Boundary",
            hovertemplate="<b>PHASE BOUNDARY</b><br>ΔG ≈ 0<extra></extra>"
        ))

    # Cross-section plane
    if show_slice:
        # Plane: Co + Cr + Fe = 1 - slice_ni  =>  Fe = (1 - slice_ni) - Co - Cr
        plane_res = 30
        p_co = np.linspace(0, 1-slice_ni, plane_res)
        p_cr = np.linspace(0, 1-slice_ni, plane_res)
        PCO, PCR = np.meshgrid(p_co, p_cr)
        PFE = (1 - slice_ni) - PCO - PCR
        valid_plane = PFE >= 0
        PCO, PCR, PFE = PCO[valid_plane], PCR[valid_plane], PFE[valid_plane]
        fig.add_trace(go.Scatter3d(
            x=PCO, y=PCR, z=PFE,
            mode="markers",
            marker=dict(size=2, color="gray", opacity=0.15, symbol="square"),
            name=f"Slice x_Ni={slice_ni:.2f}",
            hoverinfo="skip"
        ))

    # Simplex frame
    if show_simplex:
        edges = [
            [(1,0,0),(0,1,0)], [(1,0,0),(0,0,1)], [(1,0,0),(0,0,0)],
            [(0,1,0),(0,0,1)], [(0,1,0),(0,0,0)], [(0,0,1),(0,0,0)]
        ]
        for e in edges:
            fig.add_trace(go.Scatter3d(
                x=[e[0][0], e[1][0]], y=[e[0][1], e[1][1]], z=[e[0][2], e[1][2]],
                mode="lines", line=dict(color="black", width=3),
                hoverinfo="skip", showlegend=False
            ))
        vertices = [(1,0,0,"Co"), (0,1,0,"Cr"), (0,0,1,"Fe"), (0,0,0,"Ni")]
        for vx, vy, vz, vl in vertices:
            fig.add_trace(go.Scatter3d(
                x=[vx], y=[vy], z=[vz], mode="text", text=[vl],
                textposition="top center", textfont=dict(size=14, color="black"),
                hoverinfo="skip", showlegend=False
            ))

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"

# ------------------------------------------------------------------
# MODE 2: DUAL SH SURFACES (TEMPERATURE MORPH)
# ------------------------------------------------------------------
elif render_mode == "Dual SH Surfaces (Temperature Morph)" and SCIPY_AVAILABLE:
    TH, PH, G_stable, dG_grid, valid_mask, sphere_pts = sample_g_on_sphere(
        interp_liq, interp_fcc, sh_R_fixed, sh_n_theta, sh_n_phi
    )

    # Per-phase l_max with temperature dependence
    coeffs_liq, _ = fit_sh_coeffs(TH, PH, interp_liq(sphere_pts).reshape(TH.shape), l_max=l_max_liq)
    coeffs_fcc, _ = fit_sh_coeffs(TH, PH, interp_fcc(sphere_pts).reshape(TH.shape), l_max=l_max_fcc)

    if coeffs_liq is not None and coeffs_fcc is not None:
        G_liq_sh = reconstruct_sh_surface(TH, PH, coeffs_liq, l_max_liq)
        G_fcc_sh = reconstruct_sh_surface(TH, PH, coeffs_fcc, l_max_fcc)
        dG_sh = G_liq_sh - G_fcc_sh

        Co_base = sphere_pts[:, 0].reshape(TH.shape)
        Cr_base = sphere_pts[:, 1].reshape(TH.shape)
        Fe_base = sphere_pts[:, 2].reshape(TH.shape)
        Ni_base = 1.0 - Co_base - Cr_base - Fe_base
        liq_customdata = make_surface_customdata(Co_base, Cr_base, Fe_base, Ni_base, G_liq_sh, G_fcc_sh, dG_sh)
        fcc_customdata = make_surface_customdata(Co_base, Cr_base, Fe_base, Ni_base, G_liq_sh, G_fcc_sh, dG_sh)

        # === TEMPERATURE-DRIVEN SHAPE MORPHING ===
        R_liq = get_liquid_radius(G_liq_sh, sh_R_fixed, T_factor)
        X_liq = R_liq * np.sin(PH) * np.cos(TH)
        Y_liq = R_liq * np.sin(PH) * np.sin(TH)
        Z_liq = R_liq * np.cos(PH)
        X_liq, Y_liq, Z_liq = shifted_xyz(X_liq, Y_liq, Z_liq, surface_origin_radius)

        # LIQUID: fluid, shiny, expanded at high T
        fig.add_trace(go.Surface(
            x=X_liq, y=Y_liq, z=Z_liq,
            surfacecolor=G_liq_sh,
            colorscale=liq_cmap,
            cmin=G_LIQ_global_min, cmax=G_LIQ_global_max,
            opacity=liq_opacity,
            name=f"LIQUID (l={l_max_liq}, fluid)",
            showscale=True,
            colorbar=dict(
                title=dict(text="<b>G_LIQ<br>(J/mol)</b>", font=dict(size=colorbar_title_font, family="Arial Black")),
                tickfont=dict(size=colorbar_tick_font, family="Arial Black"),
                thickness=100, len=0.72, x=0.95, xanchor="left",
                outlinewidth=2, outlinecolor="black"
            ),
            customdata=liq_customdata,
            hovertemplate=thermo_hovertemplate("LIQUID", "G_LIQ surface"),
            # lighting=dict(ambient=0.55, diffuse=0.6, roughness=0.12, specular=0.9),
            lighting=dict(ambient=0.55, diffuse=0.6, roughness=0.12, specular=0.01, fresnel=1.0),
            lightposition=dict(x=100, y=100, z=50)
        ))

        R_fcc = get_fcc_radius(G_fcc_sh, sh_R_fixed, T_factor)
        X_fcc = R_fcc * np.sin(PH) * np.cos(TH)
        Y_fcc = R_fcc * np.sin(PH) * np.sin(TH)
        Z_fcc = R_fcc * np.cos(PH)
        X_fcc, Y_fcc, Z_fcc = shifted_xyz(X_fcc, Y_fcc, Z_fcc, surface_origin_radius)

        # FCC: crystalline, matte, faceted at low T
        fig.add_trace(go.Surface(
            x=X_fcc, y=Y_fcc, z=Z_fcc,
            surfacecolor=G_fcc_sh,
            colorscale=fcc_cmap,
            cmin=G_FCC_global_min, cmax=G_FCC_global_max,
            opacity=fcc_opacity,
            name=f"FCC (l={l_max_fcc}, crystal)",
            showscale=True,
            colorbar=dict(
                title=dict(text="<b>G_FCC<br>(J/mol)</b>", font=dict(size=colorbar_title_font, family="Arial Black")),
                tickfont=dict(size=colorbar_tick_font, family="Arial Black"),
                thickness=100, len=0.72, x=1.10, xanchor="left",
                outlinewidth=2, outlinecolor="black"
            ),
            customdata=fcc_customdata,
            hovertemplate=thermo_hovertemplate("FCC", "G_FCC surface"),
            # Internal FCC contour/grid lines intentionally removed for a cleaner surface.
            # lighting=dict(ambient=0.65, diffuse=0.4, roughness=0.78, specular=0.15)
            lighting=dict(ambient=0.55, diffuse=0.6, roughness=0.12, specular=0.01, fresnel=1.0),
        ))

        # ΔG = 0 contour
        if show_dg_contour:
            cx, cy, cz = extract_dg_zero_contour(TH, PH, dG_grid, sh_R_fixed)
            if len(cx) > 10:
                cx, cy, cz = shifted_xyz(cx, cy, cz, surface_origin_radius)
                fig.add_trace(go.Scatter3d(
                    x=cx, y=cy, z=cz,
                    mode="lines+markers",
                    line=dict(color=PHASE_COLORS["BOUNDARY"], width=5),
                    marker=dict(size=3, color="#f39c12", symbol="diamond"),
                    name="ΔG = 0 Transition",
                    hovertemplate="<b>PHASE BOUNDARY</b><br>ΔG ≈ 0<extra></extra>"
                ))

        # Data coverage overlay (small dots on sphere where data exists)
        if show_data_density:
            df_T = df[df["T"] == T_val]
            if len(df_T) > 0:
                fig.add_trace(go.Scatter3d(
                    x=df_T["Co"] + radial_origin_offset(surface_origin_radius)[0],
                    y=df_T["Cr"] + radial_origin_offset(surface_origin_radius)[1],
                    z=df_T["Fe"] + radial_origin_offset(surface_origin_radius)[2],
                    mode="markers",
                    marker=dict(size=3, color="black", symbol="cross", opacity=0.4),
                    name="Data Points",
                    hovertemplate="Data: Co=%{x:.3f} Cr=%{y:.3f} Fe=%{z:.3f}<extra></extra>"
                ))
    else:
        st.warning("SH fitting failed. Try adjusting l_max or resolution.")

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"

# ------------------------------------------------------------------
# MODE 3: ΔG DIFFERENCE SURFACE
# ------------------------------------------------------------------
elif render_mode == "ΔG Difference Surface" and SCIPY_AVAILABLE:
    TH, PH, G_stable, dG_grid, valid_mask, sphere_pts = sample_g_on_sphere(
        interp_liq, interp_fcc, sh_R_fixed, sh_n_theta, sh_n_phi
    )

    coeffs_dG, l_max = fit_sh_coeffs(TH, PH, dG_grid, l_max=sh_l_max)
    if coeffs_dG is not None:
        dG_smooth = reconstruct_sh_surface(TH, PH, coeffs_dG, l_max)
        G_liq_raw = interp_liq(sphere_pts).reshape(TH.shape)
        G_fcc_raw = interp_fcc(sphere_pts).reshape(TH.shape)
        Co_base = sphere_pts[:, 0].reshape(TH.shape)
        Cr_base = sphere_pts[:, 1].reshape(TH.shape)
        Fe_base = sphere_pts[:, 2].reshape(TH.shape)
        Ni_base = 1.0 - Co_base - Cr_base - Fe_base
        dg_customdata = make_surface_customdata(Co_base, Cr_base, Fe_base, Ni_base, G_liq_raw, G_fcc_raw, dG_smooth)

        # Temperature-modulated deformation
        T_deform = 1.0 + 0.2 * T_factor
        radius = sh_R_fixed * T_deform + dg_scale * dG_smooth
        radius = np.clip(radius, 0.1, 2.0)

        X = radius * np.sin(PH) * np.cos(TH)
        Y = radius * np.sin(PH) * np.sin(TH)
        Z = radius * np.cos(PH)
        X, Y, Z = shifted_xyz(X, Y, Z, surface_origin_radius)

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=dG_smooth,
            colorscale=dg_cmap,
            cmin=-dG_global_abs_max, cmax=dG_global_abs_max,
            opacity=dg_opacity,
            name="ΔG Surface",
            colorbar=dict(
                title=dict(text="<b>ΔG = G_LIQ - G_FCC<br>(J/mol)</b>", font=dict(size=colorbar_title_font, family="Arial Black")),
                tickfont=dict(size=colorbar_tick_font, family="Arial Black"),
                thickness=28, len=0.76,
                outlinewidth=2, outlinecolor="black"
            ),
            customdata=dg_customdata,
            hovertemplate=thermo_hovertemplate("ΔG Surface", "ΔG surface"),
            contours=dict(
                z=dict(show=True, highlightcolor=PHASE_COLORS["BOUNDARY"], highlightwidth=3,
                       project=dict(z=True), usecolormap=False, color=PHASE_COLORS["BOUNDARY"])
            )
        ))

        if show_dg_contour:
            cx, cy, cz = extract_dg_zero_contour(TH, PH, dG_grid, sh_R_fixed)
            if len(cx) > 10:
                fig.add_trace(go.Scatter3d(
                    x=cx, y=cy, z=cz,
                    mode="lines+markers",
                    line=dict(color=PHASE_COLORS["BOUNDARY"], width=5),
                    marker=dict(size=4, color="#f39c12", symbol="diamond"),
                    name="ΔG = 0",
                    hovertemplate="<b>BOUNDARY</b><br>ΔG ≈ 0<extra></extra>"
                ))

        st.info("""
        **Reading the ΔG Surface:**  
        🔴 **Red / dented inward** → LIQUID stable (negative ΔG)  
        🔵 **Blue / bulged outward** → FCC stable (positive ΔG)  
        🟡 **Gold contour** → ΔG = 0 phase boundary  
        Deformation amplitude = driving force magnitude.
        """)
    else:
        st.warning("SH fitting failed for ΔG")

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"

# ------------------------------------------------------------------
# MODE 4: TERNARY FLAT PROJECTION
# ------------------------------------------------------------------
elif render_mode == "Ternary Flat Projection":
    pts = generate_tetrahedral_grid(40)
    G_liq = interp_liq(pts)
    G_fcc = interp_fcc(pts)
    valid = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
    pts = pts[valid]
    G_liq = G_liq[valid]
    G_fcc = G_fcc[valid]
    dG = G_liq - G_fcc
    stable = np.where(dG <= 0, "LIQUID", "FCC")
    z_data = 1.0 - pts[:, 0] - pts[:, 1] - pts[:, 2]

    if flat_color_by == "Stable Phase":
        colors = [PHASE_COLORS[p] for p in stable]
        show_cbar = False
    elif flat_color_by == "ΔG (diverging)":
        colors = dG
        show_cbar = True
        cbar_title = "ΔG (J/mol)"
        cmin, cmax = -dG_global_abs_max, dG_global_abs_max
    elif flat_color_by == "G_magnitude":
        colors = np.minimum(G_liq, G_fcc)
        show_cbar = True
        cbar_title = "G_stable (J/mol)"
        cmin, cmax = G_global_min, G_global_max
    else:  # Data Proximity
        if HULL_AVAILABLE:
            colors = compute_data_proximity(pts, all_pts, max_dist=0.2)
        else:
            colors = np.ones(len(pts))
        show_cbar = True
        cbar_title = "Data Proximity"
        cmin, cmax = 0, 1

    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=z_data,
        mode="markers",
        marker=dict(
            size=flat_marker_size,
            color=colors,
            colorscale="RdBu_r" if flat_color_by == "ΔG (diverging)" else None,
            cmin=cmin if show_cbar else None,
            cmax=cmax if show_cbar else None,
            opacity=flat_opacity,
            symbol=[PHASE_SYMBOLS[p] for p in stable],
            line=dict(width=0.5, color="white")
        ),
        name="Ternary View",
        hovertemplate="x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Ni=%{z:.3f}<br>Phase=%{text}<extra></extra>",
        text=stable
    ))

    if show_ternary_grid:
        for ni in [0.0, 0.25, 0.5, 0.75]:
            mask = np.abs(z_data - ni) < 0.02
            if mask.sum() > 10:
                fig.add_trace(go.Scatter3d(
                    x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
                    mode="markers", marker=dict(size=1, color="gray", opacity=0.3),
                    hoverinfo="skip", showlegend=False
                ))

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Ni</sub>"

# ------------------------------------------------------------------
# MODE 5: MARKERS (DISTINCT SHAPES)
# ------------------------------------------------------------------
elif render_mode == "Markers (Distinct Shapes)":
    pts = generate_tetrahedral_grid(grid_res)
    G_liq = interp_liq(pts)
    G_fcc = interp_fcc(pts)
    valid = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
    pts = pts[valid]
    G_liq = G_liq[valid]
    G_fcc = G_fcc[valid]
    G_stable = np.minimum(G_liq, G_fcc)
    stable = np.where(G_liq <= G_fcc, "LIQUID", "FCC")
    dG = G_liq - G_fcc

    if show_uncertainty and HULL_AVAILABLE:
        proximity = compute_data_proximity(pts, all_pts, max_dist=0.2)
    else:
        proximity = np.ones(len(pts))

    if show_phase == "Stable Phase Only":
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=G_stable,
                colorscale=cmap,
                opacity=opacity * proximity,
                line=dict(width=1, color="white")
            ),
            name="Stable Phase",
            hovertemplate="<b>%{text}</b><br>G=%{marker.color:,.0f} J/mol<extra></extra>",
            text=stable
        ))
    else:
        for phase in ["LIQUID", "FCC"]:
            if phase == "LIQUID":
                mask = G_liq <= G_fcc
                g_vals = G_liq[mask]
            else:
                mask = G_fcc < G_liq
                g_vals = G_fcc[mask]

            if mask.sum() == 0:
                continue

            p_pts = pts[mask]
            p_prox = proximity[mask]

            fig.add_trace(go.Scatter3d(
                x=p_pts[:, 0], y=p_pts[:, 1], z=p_pts[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=PHASE_COLORS[phase],
                    symbol=PHASE_SYMBOLS[phase],
                    opacity=opacity * p_prox,
                    line=dict(width=1, color="white")
                ),
                name=f"{phase} Phase",
                hovertemplate=(f"<b>{phase}</b><br>" +
                               "x_Co=%{x:.3f}<br>x_Cr=%{y:.3f}<br>x_Fe=%{z:.3f}<br>" +
                               f"G={phase}=%{{marker.color:,.0f}} J/mol<extra></extra>"),
                customdata=g_vals
            ))

        if show_boundary:
            boundary_mask = np.abs(dG) < 100
            if boundary_mask.sum() > 0:
                fig.add_trace(go.Scatter3d(
                    x=pts[boundary_mask, 0], y=pts[boundary_mask, 1], z=pts[boundary_mask, 2],
                    mode="markers",
                    marker=dict(size=6, color=PHASE_COLORS["BOUNDARY"], symbol="x",
                                line=dict(width=2, color="#b7950b")),
                    name="ΔG ≈ 0 Boundary",
                    hovertemplate="<b>BOUNDARY</b><br>ΔG ≈ 0<extra></extra>"
                ))

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"

# ------------------------------------------------------------------
# MODE 6: ANIMATED TEMPERATURE SWEEP
# ------------------------------------------------------------------
elif render_mode == "Animated Temperature Sweep" and SCIPY_AVAILABLE:
    # Generate frames across temperature range
    T_frames = np.linspace(anim_start, anim_end, anim_frames)
    T_frames = [T_list[np.argmin(np.abs(np.array(T_list) - t))] for t in T_frames]
    T_frames = sorted(list(set(T_frames)))

    if len(T_frames) < 2:
        st.warning("Need at least 2 distinct temperatures for animation")
    else:
        frames = []
        for T_frame in T_frames:
            interp_liq_f, interp_fcc_f = build_interpolators_for_T(df, T_frame)
            if interp_liq_f is None:
                continue
            T_f = (T_frame - T_min) / T_range if T_range > 0 else 0.5

            TH, PH, _, dG_grid, _, sphere_pts = sample_g_on_sphere(
                interp_liq_f, interp_fcc_f, sh_R_fixed, sh_n_theta, sh_n_phi
            )

            l_max_liq = max(1, int(sh_l_max - 1.5 * T_f))
            l_max_fcc = max(2, int(sh_l_max + 1.0 * (1.0 - T_f)))

            coeffs_liq, _ = fit_sh_coeffs(TH, PH, interp_liq_f(sphere_pts).reshape(TH.shape), l_max=l_max_liq)
            coeffs_fcc, _ = fit_sh_coeffs(TH, PH, interp_fcc_f(sphere_pts).reshape(TH.shape), l_max=l_max_fcc)

            if coeffs_liq is None or coeffs_fcc is None:
                continue

            G_liq_sh = reconstruct_sh_surface(TH, PH, coeffs_liq, l_max_liq)
            G_fcc_sh = reconstruct_sh_surface(TH, PH, coeffs_fcc, l_max_fcc)

            R_liq = get_liquid_radius(G_liq_sh, sh_R_fixed, T_f)
            X_liq = R_liq * np.sin(PH) * np.cos(TH)
            Y_liq = R_liq * np.sin(PH) * np.sin(TH)
            Z_liq = R_liq * np.cos(PH)
            X_liq, Y_liq, Z_liq = shifted_xyz(X_liq, Y_liq, Z_liq, surface_origin_radius)

            R_fcc = get_fcc_radius(G_fcc_sh, sh_R_fixed, T_f)
            X_fcc = R_fcc * np.sin(PH) * np.cos(TH)
            Y_fcc = R_fcc * np.sin(PH) * np.sin(TH)
            Z_fcc = R_fcc * np.cos(PH)
            X_fcc, Y_fcc, Z_fcc = shifted_xyz(X_fcc, Y_fcc, Z_fcc, surface_origin_radius)

            frame_data = [
                go.Surface(x=X_liq, y=Y_liq, z=Z_liq, surfacecolor=G_liq_sh,
                           colorscale=liq_cmap, cmin=G_global_min, cmax=G_global_max,
                           opacity=0.6 + 0.3 * T_f, name="LIQUID", showscale=False),
                go.Surface(x=X_fcc, y=Y_fcc, z=Z_fcc, surfacecolor=G_fcc_sh,
                           colorscale=fcc_cmap, cmin=G_global_min, cmax=G_global_max,
                           opacity=0.8 - 0.4 * T_f, name="FCC", showscale=False)
            ]

            frames.append(go.Frame(data=frame_data, name=f"T={T_frame}"))

        if len(frames) > 0:
            # Initial frame
            T_init = T_frames[0]
            interp_liq_i, interp_fcc_i = build_interpolators_for_T(df, T_init)
            TH, PH, _, _, _, sphere_pts = sample_g_on_sphere(
                interp_liq_i, interp_fcc_i, sh_R_fixed, sh_n_theta, sh_n_phi
            )
            T_i = (T_init - T_min) / T_range if T_range > 0 else 0.5
            l_max_liq = max(1, int(sh_l_max - 1.5 * T_i))
            l_max_fcc = max(2, int(sh_l_max + 1.0 * (1.0 - T_i)))
            coeffs_liq, _ = fit_sh_coeffs(TH, PH, interp_liq_i(sphere_pts).reshape(TH.shape), l_max=l_max_liq)
            coeffs_fcc, _ = fit_sh_coeffs(TH, PH, interp_fcc_i(sphere_pts).reshape(TH.shape), l_max=l_max_fcc)
            G_liq_sh = reconstruct_sh_surface(TH, PH, coeffs_liq, l_max_liq)
            G_fcc_sh = reconstruct_sh_surface(TH, PH, coeffs_fcc, l_max_fcc)
            R_liq = get_liquid_radius(G_liq_sh, sh_R_fixed, T_i)
            R_fcc = get_fcc_radius(G_fcc_sh, sh_R_fixed, T_i)

            fig.add_trace(go.Surface(
                x=shifted_xyz(R_liq*np.sin(PH)*np.cos(TH), R_liq*np.sin(PH)*np.sin(TH), R_liq*np.cos(PH), surface_origin_radius)[0],
                y=shifted_xyz(R_liq*np.sin(PH)*np.cos(TH), R_liq*np.sin(PH)*np.sin(TH), R_liq*np.cos(PH), surface_origin_radius)[1],
                z=shifted_xyz(R_liq*np.sin(PH)*np.cos(TH), R_liq*np.sin(PH)*np.sin(TH), R_liq*np.cos(PH), surface_origin_radius)[2],
                surfacecolor=G_liq_sh, colorscale=liq_cmap, cmin=G_global_min, cmax=G_global_max,
                opacity=0.6+0.3*T_i, name="LIQUID", showscale=False
            ))
            fig.add_trace(go.Surface(
                x=shifted_xyz(R_fcc*np.sin(PH)*np.cos(TH), R_fcc*np.sin(PH)*np.sin(TH), R_fcc*np.cos(PH), surface_origin_radius)[0],
                y=shifted_xyz(R_fcc*np.sin(PH)*np.cos(TH), R_fcc*np.sin(PH)*np.sin(TH), R_fcc*np.cos(PH), surface_origin_radius)[1],
                z=shifted_xyz(R_fcc*np.sin(PH)*np.cos(TH), R_fcc*np.sin(PH)*np.sin(TH), R_fcc*np.cos(PH), surface_origin_radius)[2],
                surfacecolor=G_fcc_sh, colorscale=fcc_cmap, cmin=G_global_min, cmax=G_global_max,
                opacity=0.8-0.4*T_i, name="FCC", showscale=False
            ))

            fig.frames = frames

            # Animation controls
            fig.update_layout(
                updatemenus=[{
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "▶️ Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 800, "redraw": True},
                                            "fromcurrent": True, "transition": {"duration": 300}}]
                        },
                        {
                            "label": "⏸️ Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                              "mode": "immediate", "transition": {"duration": 0}}]
                        }
                    ],
                    "x": 0.1, "y": 0.05
                }],
                sliders=[{
                    "active": 0,
                    "yanchor": "top", "xanchor": "left",
                    "currentvalue": {"prefix": "Temperature: ", "visible": True, "xanchor": "right"},
                    "transition": {"duration": 300},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9, "x": 0.1, "y": 0,
                    "steps": [
                        {"method": "animate", "args": [[f"T={T_f}"], {"frame": {"duration": 300, "redraw": True},
                                                                      "mode": "immediate", "transition": {"duration": 300}}],
                         "label": f"{T_f}K"} for T_f in T_frames
                    ]
                }]
            )

            st.info(f"Animation ready: {len(frames)} frames from {anim_start}K to {anim_end}K. Click **▶️ Play** or drag the slider.")
        else:
            st.error("Could not generate animation frames.")

    scene_x, scene_y, scene_z = "x<sub>Co</sub>", "x<sub>Cr</sub>", "x<sub>Fe</sub>"

# ------------------------------------------------------------------
# COMMON OVERLAYS
# ------------------------------------------------------------------

# Composition path (connects query history)
if show_comp_path and len(st.session_state.query_history) > 1:
    hist = st.session_state.query_history
    path_x = [h["Co"] for h in hist]
    path_y = [h["Cr"] for h in hist]
    path_z = [h["Fe"] for h in hist]
    path_T = [h["T"] for h in hist]

    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode="lines+markers",
        line=dict(color="gold", width=4, dash="dot"),
        marker=dict(size=6, color=path_T, colorscale="Thermal", cmin=T_min, cmax=T_max,
                    showscale=True, colorbar=dict(title="Path T (K)", thickness=15, len=0.5)),
        name="Composition Path",
        hovertemplate="T=%{marker.color:.0f}K<br>Co=%{x:.3f}<br>Cr=%{y:.3f}<br>Fe=%{z:.3f}<extra></extra>"
    ))

# Query point
if query_result is not None:
    q_color = PHASE_COLORS[query_result["Phase"]]
    q_symbol = PHASE_SYMBOLS[query_result["Phase"]]

    fig.add_trace(go.Scatter3d(
        x=[query_result["Co"]], y=[query_result["Cr"]], z=[query_result["Fe"]],
        mode="markers+text",
        marker=dict(size=18, color=q_color, symbol=q_symbol,
                    line=dict(width=3, color="white")),
        text=["QUERY"],
        textposition="top center",
        textfont=dict(size=12, color=q_color, family="Arial Black"),
        name=f"Query ({query_result['Phase']})",
        hovertemplate=(f"<b>QUERY</b><br>T={query_result['T']}K<br>" +
                       f"Co={query_result['Co']:.3f}<br>Cr={query_result['Cr']:.3f}<br>" +
                       f"Fe={query_result['Fe']:.3f}<br>Ni={query_result['Ni']:.3f}<br>" +
                       f"G_LIQ={query_result['G_LIQ']:,.0f} J/mol<br>" +
                       f"G_FCC={query_result['G_FCC']:,.0f} J/mol<br>" +
                       f"G_stable={query_result['G_stable']:,.0f} J/mol<br>" +
                       f"ΔG={query_result['dG']:,.0f} J/mol<br>" +
                       f"Phase={query_result['Phase']}<extra></extra>")
    ))

    if show_query_probe:
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        r_probe = 0.08
        x_p = query_result["Co"] + r_probe * np.outer(np.cos(u), np.sin(v))
        y_p = query_result["Cr"] + r_probe * np.outer(np.sin(u), np.sin(v))
        z_p = query_result["Fe"] + r_probe * np.outer(np.ones(np.size(u)), np.cos(v))
        fig.add_trace(go.Surface(
            x=x_p, y=y_p, z=z_p,
            opacity=0.2,
            colorscale=[[0, q_color], [1, q_color]],
            showscale=False,
            name="Query Probe",
            hoverinfo="skip"
        ))

# Coordinate axes
if show_axes_frame:
    axis_len = 1.05
    ax_ox, ax_oy, ax_oz = radial_origin_offset(axes_origin_radius)

    for coord, color, label in [(0, "#c0392b", "Co"), (1, "#27ae60", "Cr"), (2, "#2980b9", "Fe")]:
        x_line = [ax_ox, ax_ox + (axis_len if coord == 0 else 0)]
        y_line = [ax_oy, ax_oy + (axis_len if coord == 1 else 0)]
        z_line = [ax_oz, ax_oz + (axis_len if coord == 2 else 0)]

        fig.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode="lines+text",
            line=dict(color=color, width=coord_axis_line_width),
            text=["", f"<b>{label}</b>"],
            textposition="top center",
            textfont=dict(size=coord_axis_label_font, color=color, family="Arial Black"),
            hoverinfo="skip", showlegend=False
        ))

# ------------------------------------------------------------------
# CLIENT-SIDE PLOT RENDER + CURRENT VIEW PNG DOWNLOAD
# ------------------------------------------------------------------
def render_plotly_with_current_view_download(fig, plot_width, plot_height, filename_base):
    """Render Plotly figure in an HTML component and download the current camera view client-side."""
    plot_div_id = f"plot_{uuid.uuid4().hex}"
    button_id = f"btn_{uuid.uuid4().hex}"
    fig_json = fig.to_json()

    html = f"""
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <div style="display:flex; justify-content:flex-end; margin-bottom:10px;">
        <button id="{button_id}" style="
            background-color:#ffffff;
            color:#111111;
            border:2px solid #111111;
            border-radius:8px;
            padding:8px 14px;
            font-size:15px;
            font-weight:700;
            cursor:pointer;
        ">🖼️ Download Transparent PNG (Current View)</button>
    </div>
    <div id="{plot_div_id}" style="width:{plot_width}px; height:{plot_height}px;"></div>
    <script>
        const fig = {fig_json};
        const gd = document.getElementById("{plot_div_id}");
        Plotly.newPlot(gd, fig.data, fig.layout, {{
            responsive: true,
            displaylogo: false,
            toImageButtonOptions: {{
                format: 'png',
                filename: '{filename_base}',
                width: {plot_width},
                height: {plot_height},
                scale: 3
            }}
        }});

        document.getElementById("{button_id}").addEventListener('click', function() {{
            Plotly.downloadImage(gd, {{
                format: 'png',
                filename: '{filename_base}',
                width: {plot_width},
                height: {plot_height},
                scale: 3
            }});
        }});
    </script>
    """

    components.html(html, height=plot_height + 70, width=plot_width + 20)

# ------------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------------
def make_axis(title_text):
    tick_vals = [0.00, 0.25, 0.50, 0.75, 1.00]
    tick_text = ["0.00", "0.25", "0.50", "0.75", "1.00"]
    return dict(
        title=dict(text=""),
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        ticks="outside",
        ticklen=axis_tick_length,
        tickwidth=3,
        tickcolor="black",
        tickfont=dict(size=axis_tick_font, family="Arial Black", color="black"),
        showbackground=True,
        backgroundcolor="rgba(0,0,0,0)",
        gridcolor="rgba(80,80,80,0.30)",
        gridwidth=2,
        zerolinecolor="rgba(30,30,30,0.75)",
        zerolinewidth=3,
        showline=True,
        linewidth=4,
        linecolor="black"
    )

fig.update_layout(
    template=template,
    scene=dict(
        xaxis=make_axis(scene_x),
        yaxis=make_axis(scene_y),
        zaxis=make_axis(scene_z),
        aspectmode="cube",
        bgcolor="rgba(0,0,0,0)",
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.1))
    ),
    title=dict(
        text=f"Co-Cr-Fe-Ni at T = {T_val} K | {render_mode} | {phase_expected}",
        font=dict(size=title_font, family="Arial Black", color="black")
    ),
    font=dict(size=18, family="Arial Black"),
    width=plot_width,
    height=plot_height,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=190, b=60 if render_mode=="Animated Temperature Sweep" else 0, t=70),
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.01,
        bgcolor="rgba(255,255,255,0.88)", bordercolor="black", borderwidth=2,
        font=dict(size=16, family="Arial Black")
    )
)

try:
    render_plotly_with_current_view_download(
        fig,
        plot_width=plot_width,
        plot_height=plot_height,
        filename_base=f"CoCrFeNi_T{T_val}K_transparent_current_view"
    )
    st.caption("The transparent PNG download button above saves the figure in the current camera orientation shown on screen.")
except Exception as e:
    st.error(f"Render error: {e}")

# =============================================
# EXPORT & FOOTER
# =============================================
with st.expander("💾 Export & Data", expanded=False):
    col1, col2, col3 = st.columns(3)

    # Export current view data
    if render_mode in ["Phase Boundary (Scientific)", "Markers (Distinct Shapes)", "Ternary Flat Projection"]:
        pts = generate_tetrahedral_grid(35)
        G_liq = interp_liq(pts)
        G_fcc = interp_fcc(pts)
        valid = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
        export_df = pd.DataFrame({
            "Co": pts[valid, 0], "Cr": pts[valid, 1], "Fe": pts[valid, 2],
            "Ni": 1.0 - pts[valid, 0] - pts[valid, 1] - pts[valid, 2],
            "G_LIQ": G_liq[valid], "G_FCC": G_fcc[valid],
            "dG": G_liq[valid] - G_fcc[valid],
            "Stable_Phase": np.where(G_liq[valid] <= G_fcc[valid], "LIQUID", "FCC"),
            "T": T_val
        })
        csv = export_df.to_csv(index=False)
        col1.download_button("📥 Download CSV", csv, f"CoCrFeNi_T{T_val}K.csv", "text/csv")

    # Export figure as HTML
    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
    col2.download_button("🌐 Download HTML", html_str, f"CoCrFeNi_T{T_val}K.html", "text/html")

    # Export figure as transparent PNG
    col3.info("Use the 'Download Transparent PNG (Current View)' button above the plot. It saves the exact camera orientation currently shown on screen with transparent background.")

    # Query history
    if len(st.session_state.query_history) > 0:
        st.subheader("Query History")
        hist_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(hist_df.style.format({
            "Co": "{:.3f}", "Cr": "{:.3f}", "Fe": "{:.3f}", "Ni": "{:.3f}",
            "G_LIQ": "{:.0f}", "G_FCC": "{:.0f}", "G_stable": "{:.0f}", "dG": "{:.0f}"
        }), width='stretch')
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()

with st.expander("📖 How to Read Each Mode", expanded=True):
    st.markdown("""
    ### Phase Boundary (Scientific) — **Most Accurate**
    True tetrahedral composition space. 🔴 circles = LIQUID, 🔵 diamonds = FCC, 🟡 X's = ΔG≈0 boundary.  
    **Uncertainty fading**: points far from data are translucent. **Cross-section plane**: slice at fixed Ni.

    ### Dual SH Surfaces (Temperature Morph) — **Aesthetic + Physical**
    Two SH surfaces with **temperature-driven shape morphing**:
    - **LIQUID** (Red): Becomes **larger, smoother, shinier** at high T (fluid expansion)
    - **FCC** (Blue): Becomes **smaller, faceted, matte** at low T (crystalline ripples)
    - Auto l_max: LIQUID uses lower l at high T (smooth); FCC uses higher l at low T (faceted)
    - 🟡 Gold line = ΔG = 0 intersection

    ### ΔG Difference Surface
    Single sphere deformed by ΔG. **Red/dented** = LIQUID, **Blue/bulged** = FCC. Amplitude = driving force.

    ### Ternary Flat Projection
    Standard materials view: x=Co, y=Cr, z=Ni. Shape = phase, color = ΔG or proximity.

    ### Markers (Distinct Shapes)
    Classic 3D scatter with **circle vs diamond** per phase. Boundary points in gold.

    ### Animated Temperature Sweep
    Play button morphs between temperatures. Watch LIQUID grow and FCC shrink as T increases.
    """)