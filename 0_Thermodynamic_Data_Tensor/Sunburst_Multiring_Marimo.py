# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.10.0",
#   "numpy>=1.24",
#   "pandas>=2.0",
#   "plotly>=5.20",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full", app_title="CoCrFeNi Multi-Ring Chart")


@app.cell(hide_code=True)
def _():
    import os
    import io
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import marimo as mo

    return go, io, mo, np, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🌞 CoCrFeNi Temperature–Composition–Interface Driving Force Multi-Ring Chart

        This marimo app loads the same `Gibbs_<T>K.csv` thermodynamic files used by the 8D parallel-plot app,
        then builds the temperature–composition–force multi-ring Plotly chart.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    csv_folder = mo.ui.text(
        value="https://subediupadesh.github.io/laserheattensor/csv_files/",
        label="CSV folder",
    )

    T_min_input = mo.ui.number(value=300, step=100, label="Minimum temperature [K]")
    T_max_input = mo.ui.number(value=3300, step=100, label="Maximum temperature [K]")
    T_step_input = mo.ui.number(value=100, step=100, label="Temperature step [K]")

    area_mode = mo.ui.dropdown(
        options=["Grain Size Derived (Sv x V)", "Direct Input (A)"],
        value="Grain Size Derived (Sv x V)",
        label="Area mode",
    )

    temp_cmap = mo.ui.dropdown(
        options=[
            "thermal",
            "inferno",
            "magma",
            "plasma",
            "viridis",
            "Plotly3",
            "Portland",
            "Bluered",
            "cividis",
            "blackbody",
            "hot",
            "turbo",
            "temps",
        ],
        value="Plotly3",
        label="Temperature colorscale",
    )

    force_cmap = mo.ui.dropdown(
        options=[
            "Portland_r",
            "Portland",
            "Plotly3",
            "ice",
            "haline",
            "deep",
            "dense",
            "teal",
            "tealgrn",
            "blues",
            "blugrn",
            "pubu",
            "electric",
        ],
        value="Portland_r",
        label="Force colorscale",
    )

    sb_sample = mo.ui.slider(
        start=3,
        stop=12,
        step=1,
        value=5,
        label="Compositions per temperature",
    )

    sample_seed = mo.ui.number(value=42, step=1, label="Random seed")

    grain_size_um = mo.ui.number(
        start=0.001,
        stop=10000.0,
        step=0.1,
        value=2.5,
        label="Average grain size d [μm]",
    )

    shape_choice = mo.ui.dropdown(
        options=["Spherical (k=2)", "Tetrakaidecahedron (k=3)", "Equiaxed cubic (k=6)"],
        value="Tetrakaidecahedron (k=3)",
        label="Grain shape factor",
    )

    sample_volume_cm3 = mo.ui.number(
        start=1e-9,
        stop=1e6,
        step=0.1,
        value=1.0,
        label="Sample volume V [cm³]",
    )

    dV_um3 = mo.ui.number(
        start=0.001,
        stop=1000.0,
        step=0.1,
        value=1.0,
        label="Local volume dV [μm³]",
    )

    gamma_input = mo.ui.number(
        start=0.01,
        stop=5.0,
        step=0.01,
        value=0.6,
        label="γ Liquid/FCC [N/m]",
    )

    use_capillary = mo.ui.checkbox(
        value=True,
        label="Enable capillary correction: P_net = P_chem − 2γ/r",
    )

    sb_area_input = mo.ui.number(
        start=1e-12,
        stop=1e2,
        step=1e-10,
        value=1e-8,
        label="Reference area A [m²]",
    )

    show_raw_data = mo.ui.checkbox(value=False, label="Show loaded dataframe preview")
    show_debug = mo.ui.checkbox(value=False, label="Show debug summary")

    sidebar = mo.sidebar(
        mo.vstack(
            [
                mo.md("## Data settings"),
                csv_folder,
                T_min_input,
                T_max_input,
                T_step_input,
                mo.md("## Chart settings"),
                area_mode,
                temp_cmap,
                force_cmap,
                sb_sample,
                sample_seed,
                mo.md("## Grain-size / force settings"),
                grain_size_um,
                shape_choice,
                sample_volume_cm3,
                dV_um3,
                gamma_input,
                use_capillary,
                sb_area_input,
                mo.md("## Display"),
                show_raw_data,
                show_debug,
            ],
            gap=1,
        )
    )

    sidebar

    return (
        T_max_input,
        T_min_input,
        T_step_input,
        area_mode,
        csv_folder,
        dV_um3,
        force_cmap,
        gamma_input,
        grain_size_um,
        sample_seed,
        sample_volume_cm3,
        sb_area_input,
        sb_sample,
        shape_choice,
        show_debug,
        show_raw_data,
        temp_cmap,
        use_capillary,
    )


@app.cell(hide_code=True)
def _(io, np, os, pd):
    REQUIRED_COLUMNS = ["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"]

    def safe_numeric(value, fallback):
        try:
            if value is None:
                return fallback
            if isinstance(value, str) and value.strip() == "":
                return fallback
            return value
        except Exception:
            return fallback

    def clean_gibbs_dataframe(df):
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        df = df[REQUIRED_COLUMNS].copy()

        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=REQUIRED_COLUMNS)
        df["sum_x"] = df["Co"] + df["Cr"] + df["Fe"] + df["Ni"]
        df = df[np.abs(df["sum_x"] - 1.0) < 1e-6].copy()

        if df.empty:
            raise ValueError("No valid rows after enforcing Co + Cr + Fe + Ni = 1.")

        return df

    def load_temperature_data(csv_folder_value, T_min_value, T_max_value, T_step_value):
        temperatures_to_try = list(
            range(int(T_min_value), int(T_max_value) + 1, int(T_step_value))
        )

        data_by_T = {}
        missing_files = []

        csv_folder_value = str(csv_folder_value).strip().rstrip("/")

        try:
            from pyodide.http import open_url
            running_in_wasm = True
        except ImportError:
            open_url = None
            running_in_wasm = False

        for T in temperatures_to_try:
            filename = f"Gibbs_{T}K.csv"

            if running_in_wasm:
                file_path = f"{csv_folder_value}/{filename}"

                try:
                    csv_text = open_url(file_path).read()
                    df_T = pd.read_csv(io.StringIO(csv_text))
                except Exception as exc:
                    missing_files.append(f"{file_path} [browser read error: {exc}]")
                    continue

            else:
                if csv_folder_value.startswith("http://") or csv_folder_value.startswith("https://"):
                    file_path = f"{csv_folder_value}/{filename}"
                else:
                    file_path = os.path.join(csv_folder_value, filename)

                try:
                    df_T = pd.read_csv(file_path)
                except Exception as exc:
                    missing_files.append(f"{file_path} [local/web read error: {exc}]")
                    continue

            try:
                data_by_T[T] = clean_gibbs_dataframe(df_T)
            except Exception as exc:
                missing_files.append(f"{file_path} [data validation error: {exc}]")
                continue

        temperatures = sorted(data_by_T.keys())

        if len(temperatures) == 0:
            return None, [], missing_files

        return data_by_T, temperatures, missing_files

    return clean_gibbs_dataframe, load_temperature_data, safe_numeric


@app.cell(hide_code=True)
def _(
    T_max_input,
    T_min_input,
    T_step_input,
    csv_folder,
    load_temperature_data,
    mo,
    safe_numeric,
):
    T_min_value = int(safe_numeric(T_min_input.value, 300))
    T_max_value = int(safe_numeric(T_max_input.value, 3300))
    T_step_value = int(safe_numeric(T_step_input.value, 100))

    if T_step_value <= 0:
        T_step_value = 100

    data_by_T, temperatures, missing_files = load_temperature_data(
        csv_folder.value,
        T_min_value,
        T_max_value,
        T_step_value,
    )

    if data_by_T is None:
        data_status = mo.callout(
            (
                "No valid CSV files were found. Check folder path and filename pattern: "
                f"`Gibbs_<T>K.csv`. Current folder: `{csv_folder.value}`. "
                "Open the `Missing files` section below to see browser/local read errors."
            ),
            kind="danger",
        )
        data_by_T = {}
    else:
        first_T = temperatures[0]
        total_rows = sum(len(df) for df in data_by_T.values())
        data_status = mo.callout(
            (
                f"Loaded `{len(temperatures)}` temperature files from `{csv_folder.value}` "
                f"with `{total_rows}` valid rows. Temperature range: `{min(temperatures)}–{max(temperatures)} K`. "
                f"Rows at first temperature `{first_T} K`: `{len(data_by_T[first_T])}`."
            ),
            kind="success",
        )

    missing_status = None

    if missing_files:
        preview = "\n".join([f"- `{item}`" for item in missing_files[:40]])
        extra = "" if len(missing_files) <= 40 else f"\n- ... and {len(missing_files) - 40} more"
        missing_status = mo.accordion({"Missing / skipped files": mo.md(preview + extra)})

    return data_by_T, data_status, missing_files, missing_status, temperatures


@app.cell(hide_code=True)
def _():
    PURE_VM = {
        "Co": 6.80e-6,
        "Cr": 7.23e-6,
        "Fe": 7.09e-6,
        "Ni": 6.59e-6,
    }

    T_MIN_NORMALIZE = 300
    T_MAX_NORMALIZE = 3300
    DEFAULT_DV = 1e-18

    GRAIN_SHAPE_FACTORS = {
        "Spherical (k=2)": 2.0,
        "Tetrakaidecahedron (k=3)": 3.0,
        "Equiaxed cubic (k=6)": 6.0,
    }

    def composition_dependent_vm(x_co, x_cr, x_fe, x_ni):
        return (
            x_co * PURE_VM["Co"]
            + x_cr * PURE_VM["Cr"]
            + x_fe * PURE_VM["Fe"]
            + x_ni * PURE_VM["Ni"]
        )

    def normalize_temperature(T):
        return (T - T_MIN_NORMALIZE) / (T_MAX_NORMALIZE - T_MIN_NORMALIZE)

    def compute_Sv(grain_size_m, shape_factor):
        return shape_factor / grain_size_m

    def compute_total_area(Sv, sample_volume_m3):
        return Sv * sample_volume_m3

    def compute_curvature_radius(grain_size_m):
        return grain_size_m / 4.0

    def compute_capillary_pressure(gamma, curvature_r):
        return (2.0 * gamma) / curvature_r

    def compute_net_pressure(P_chem, P_capillary):
        return P_chem - P_capillary

    def compute_differential_force(P_net, Sv, dV):
        return P_net * Sv * dV

    return (
        DEFAULT_DV,
        GRAIN_SHAPE_FACTORS,
        composition_dependent_vm,
        compute_Sv,
        compute_capillary_pressure,
        compute_curvature_radius,
        compute_differential_force,
        compute_net_pressure,
        compute_total_area,
        normalize_temperature,
    )


@app.cell(hide_code=True)
def _(
    DEFAULT_DV,
    GRAIN_SHAPE_FACTORS,
    area_mode,
    compute_Sv,
    compute_total_area,
    dV_um3,
    gamma_input,
    grain_size_um,
    sample_volume_cm3,
    sb_area_input,
    shape_choice,
    use_capillary,
):
    if area_mode.value == "Grain Size Derived (Sv x V)":
        grain_size_m = float(grain_size_um.value) * 1e-6
        shape_factor = GRAIN_SHAPE_FACTORS[shape_choice.value]
        sample_volume_m3 = float(sample_volume_cm3.value) * 1e-6
        dV = float(dV_um3.value) * 1e-18
        gamma = float(gamma_input.value)

        Sv = compute_Sv(grain_size_m, shape_factor)
        interface_area = compute_total_area(Sv, sample_volume_m3)
        sb_area = interface_area
        capillary_enabled = bool(use_capillary.value)
    else:
        grain_size_m = None
        shape_factor = None
        sample_volume_m3 = None
        dV = DEFAULT_DV
        gamma = 0.0
        Sv = None
        interface_area = float(sb_area_input.value)
        sb_area = float(sb_area_input.value)
        capillary_enabled = False

    return (
        Sv,
        capillary_enabled,
        dV,
        gamma,
        grain_size_m,
        interface_area,
        sample_volume_m3,
        sb_area,
        shape_factor,
    )


@app.cell(hide_code=True)
def _(Sv, area_mode, dV, interface_area, mo):
    if area_mode.value == "Grain Size Derived (Sv x V)":
        parameter_summary = mo.md(
            f"""
            ### Current derived quantities

            - **Sv:** `{Sv:.2e} m²/m³`
            - **A_total:** `{interface_area:.2e} m²`
            - **dV:** `{dV:.2e} m³`
            """
        )
    else:
        parameter_summary = mo.md(
            f"""
            ### Current direct-area quantity

            - **Reference area A:** `{interface_area:.2e} m²`
            """
        )

    parameter_summary
    return parameter_summary


@app.cell(hide_code=True)
def _(
    Sv,
    area_mode,
    capillary_enabled,
    composition_dependent_vm,
    compute_capillary_pressure,
    compute_curvature_radius,
    compute_differential_force,
    compute_net_pressure,
    dV,
    data_by_T,
    force_cmap,
    gamma,
    go,
    grain_size_m,
    normalize_temperature,
    np,
    sample_seed,
    sb_area,
    sb_sample,
    temp_cmap,
    temperatures,
):
    if not data_by_T:
        fig = None
        debug_summary = {}
    else:
        temperatures_sorted = sorted(temperatures)
        n_temp = len(temperatures_sorted)

        temp_ring_base = 0.00
        temp_ring_thickness = 0.85
        comp_ring_base = 1.02
        comp_ring_thickness = 1.25
        force_ring_base = 2.45
        force_ring_thickness = 0.90
        temp_width = 360.0 / n_temp

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

        comp_colors_map = {
            "Co": "#0057B8",
            "Cr": "#D62828",
            "Fe": "#2A9D3F",
            "Ni": "#F4C430",
        }

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
            "Ni": {"theta": [], "width": [], "r": [], "base": [], "custom": []},
        }

        force_theta = []
        force_widths = []
        force_r = []
        force_base = []
        force_colors = []
        force_custom = []

        rng = np.random.default_rng(int(sample_seed.value))

        for iT, T_sun in enumerate(temperatures_sorted):
            theta_start = iT * temp_width
            theta_center_temp = theta_start + temp_width / 2.0

            temp_theta.append(theta_center_temp)
            temp_widths.append(temp_width)
            temp_r.append(temp_ring_thickness)
            temp_base.append(temp_ring_base)
            temp_colors.append(float(T_sun))
            temp_custom.append([T_sun, normalize_temperature(T_sun), f"{T_sun} K"])

            df_temp = data_by_T[T_sun].copy()

            if df_temp.empty:
                continue

            n_samples = min(int(sb_sample.value), len(df_temp))

            if n_samples <= 0:
                continue

            if n_samples >= len(df_temp):
                sample_df = df_temp.sample(
                    frac=1.0,
                    random_state=int(sample_seed.value),
                ).reset_index(drop=True)
            else:
                sample_idx = rng.choice(
                    len(df_temp),
                    size=n_samples,
                    replace=False,
                )
                sample_df = df_temp.iloc[sample_idx].copy().reset_index(drop=True)

            sample_df = sample_df.sort_values(["Co", "Cr", "Fe", "Ni"]).reset_index(drop=True)
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

                V_m_s = composition_dependent_vm(x_co_s, x_cr_s, x_fe_s, x_ni_s)
                P_chem_pa_s = -delta_G_s / V_m_s
                P_chem_mpa_s = P_chem_pa_s / 1e6

                if capillary_enabled and grain_size_m is not None:
                    chart_curvature_r = compute_curvature_radius(grain_size_m)
                    chart_P_capillary = compute_capillary_pressure(gamma, chart_curvature_r)
                    P_net_s = compute_net_pressure(P_chem_pa_s, chart_P_capillary)
                    P_net_mpa_s = P_net_s / 1e6
                else:
                    P_net_s = P_chem_pa_s
                    P_net_mpa_s = P_chem_mpa_s

                if area_mode.value == "Grain Size Derived (Sv x V)" and Sv is not None:
                    net_force_s = compute_differential_force(P_net_s, Sv, dV)
                    force_mode_s = "dF_net = P_net × Sv × dV"
                else:
                    net_force_s = P_net_s * float(sb_area)
                    force_mode_s = "F = P_net × A"

                comp_vals = {
                    "Co": x_co_s,
                    "Cr": x_cr_s,
                    "Fe": x_fe_s,
                    "Ni": x_ni_s,
                }

                cumulative_base = comp_ring_base

                for el in ["Co", "Cr", "Fe", "Ni"]:
                    thickness_el = comp_vals[el] * comp_ring_thickness

                    element_data[el]["theta"].append(theta_local_center)
                    element_data[el]["width"].append(comp_width)
                    element_data[el]["r"].append(thickness_el)
                    element_data[el]["base"].append(cumulative_base)

                    element_data[el]["custom"].append(
                        [
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
                            force_mode_s,
                        ]
                    )

                    cumulative_base += thickness_el

                force_theta.append(theta_local_center)
                force_widths.append(comp_width)
                force_r.append(force_ring_thickness)
                force_base.append(force_ring_base)
                force_colors.append(net_force_s)

                force_custom.append(
                    [
                        T_sun,
                        x_co_s,
                        x_cr_s,
                        x_fe_s,
                        x_ni_s,
                        delta_G_s,
                        P_chem_mpa_s,
                        net_force_s,
                        P_net_mpa_s,
                        force_mode_s,
                    ]
                )

        if len(force_colors) > 0:
            max_abs_force = max(
                abs(float(np.nanmin(force_colors))),
                abs(float(np.nanmax(force_colors))),
            )
            if max_abs_force == 0:
                max_abs_force = 1.0
        else:
            max_abs_force = 1.0

        temp_tickvals = np.linspace(float(min(temperatures_sorted)), float(max(temperatures_sorted)), 5)
        temp_ticktext = [f"{v:.0f}" for v in temp_tickvals]

        force_tickvals = np.linspace(-float(max_abs_force), float(max_abs_force), 5)
        force_ticktext = [f"{v:.2e}" for v in force_tickvals]

        fig = go.Figure()

        fig.add_trace(
            go.Barpolar(
                theta=temp_theta,
                width=temp_widths,
                r=temp_r,
                base=temp_base,
                marker=dict(
                    color=temp_colors,
                    colorscale=temp_cmap.value,
                    cmin=min(temperatures_sorted),
                    cmax=max(temperatures_sorted),
                    colorbar=dict(
                        title=dict(
                            text="<b>Temperature<br>[K]</b>",
                            font=dict(size=COLORBAR_TITLE_SIZE, family=FONT_FAMILY, color="black"),
                        ),
                        tickfont=dict(size=COLORBAR_TICK_SIZE, family=FONT_FAMILY, color="black"),
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
                        tickformat=".0f",
                    ),
                    line=dict(color=RING_BORDER_COLOR, width=RING_BORDER_WIDTH),
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
                opacity=0.98,
            )
        )

        for el in ["Co", "Cr", "Fe", "Ni"]:
            fig.add_trace(
                go.Barpolar(
                    theta=element_data[el]["theta"],
                    width=element_data[el]["width"],
                    r=element_data[el]["r"],
                    base=element_data[el]["base"],
                    marker=dict(
                        color=comp_colors_map[el],
                        line=dict(color=COMPOSITION_BORDER_COLOR, width=COMPOSITION_BORDER_WIDTH),
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
                    opacity=0.98,
                )
            )

        fig.add_trace(
            go.Barpolar(
                theta=force_theta,
                width=force_widths,
                r=force_r,
                base=force_base,
                marker=dict(
                    color=force_colors,
                    colorscale=force_cmap.value,
                    cmin=-max_abs_force,
                    cmax=max_abs_force,
                    cmid=0,
                    colorbar=dict(
                        title=dict(
                            text="<b>Differential Force<br>[N]</b>",
                            font=dict(size=COLORBAR_TITLE_SIZE, family=FONT_FAMILY, color="black"),
                        ),
                        tickfont=dict(size=COLORBAR_TICK_SIZE, family=FONT_FAMILY, color="black"),
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
                        tickformat=".1e",
                    ),
                    line=dict(color=RING_BORDER_COLOR, width=RING_BORDER_WIDTH),
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
                opacity=0.98,
            )
        )

        fig.add_annotation(
            text="<b>INNER RING: TEMPERATURE</b>",
            x=0.02,
            y=1.065,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=17, family=FONT_FAMILY, color="black"),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="black",
            borderwidth=1.6,
            borderpad=5,
        )

        fig.add_annotation(
            text="<b>MIDDLE RING: Co / Cr / Fe / Ni COMPOSITION</b>",
            x=0.39,
            y=1.065,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=17, family=FONT_FAMILY, color="black"),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="black",
            borderwidth=1.6,
            borderpad=5,
        )

        fig.add_annotation(
            text="<b>OUTER RING: dF_net [N]</b>",
            x=0.79,
            y=1.065,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=17, family=FONT_FAMILY, color="black"),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="black",
            borderwidth=1.6,
            borderpad=5,
        )

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
                font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color="black"),
            ),
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=1080,
            width=1250,
            margin=dict(t=160, b=45, l=45, r=285),
            font=dict(family=FONT_FAMILY, size=18, color="black"),
            legend=dict(
                title=dict(
                    text="<b>Composition</b>",
                    font=dict(size=22, family=FONT_FAMILY, color="black"),
                ),
                font=dict(size=LEGEND_FONT_SIZE, family=FONT_FAMILY, color="black"),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="black",
                borderwidth=2.2,
                x=1.05,
                y=0.24,
                xanchor="left",
                yanchor="middle",
                itemsizing="constant",
            ),
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font=dict(size=HOVER_FONT_SIZE, family=FONT_FAMILY, color="black"),
            ),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=False,
                    range=[0, force_ring_base + force_ring_thickness + 0.30],
                    showline=False,
                    showgrid=False,
                ),
                angularaxis=dict(
                    direction="clockwise",
                    rotation=90,
                    showticklabels=False,
                    ticks="",
                    showline=False,
                    gridcolor="rgba(120,120,120,0.22)",
                    gridwidth=1.0,
                ),
            ),
        )

        fig.update_polars(sector=[0, 360])

        debug_summary = {
            "temperature_order": temperatures_sorted,
            "n_temperatures": n_temp,
            "temperature_sector_width_deg": temp_width,
            "n_force_segments": len(force_theta),
            "differential_force_min_N": float(np.min(force_colors)) if len(force_colors) > 0 else None,
            "differential_force_max_N": float(np.max(force_colors)) if len(force_colors) > 0 else None,
            "max_abs_differential_force_N": float(max_abs_force),
            "composition_colors": comp_colors_map,
            "ring_geometry": {
                "temperature": {"base": temp_ring_base, "thickness": temp_ring_thickness},
                "composition": {"base": comp_ring_base, "thickness": comp_ring_thickness},
                "force": {"base": force_ring_base, "thickness": force_ring_thickness},
            },
        }

    return debug_summary, fig


@app.cell(hide_code=True)
def _(fig, mo):
    if fig is None:
        interactive_plot = mo.callout("Chart cannot be generated because no valid data was loaded.", kind="danger")
    else:
        interactive_plot = mo.ui.plotly(
            fig,
            config={
                "displaylogo": False,
                "responsive": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "CoCrFeNi_temperature_composition_force_multiring_transparent",
                    "height": 2200,
                    "width": 2400,
                    "scale": 1,
                },
            },
        )

    return interactive_plot,


@app.cell(hide_code=True)
def _(
    data_by_T,
    data_status,
    interactive_plot,
    missing_status,
    mo,
    parameter_summary,
    show_raw_data,
):
    output_items = [data_status]

    if missing_status is not None:
        output_items.append(missing_status)

    output_items.extend(
        [
            parameter_summary,
            mo.md("## 🌞 Multi-Ring Mechanical Force–Thermodynamic State Space Chart"),
            mo.callout(
                "Use the Plotly toolbar camera button to download the current figure as PNG. The figure background is transparent.",
                kind="info",
            ),
            interactive_plot,
        ]
    )

    if show_raw_data.value and data_by_T:
        first_T = sorted(data_by_T.keys())[0]
        output_items.extend(
            [
                mo.md(f"## Loaded data preview at {first_T} K"),
                data_by_T[first_T].head(50),
            ]
        )

    mo.vstack(output_items, gap=1)
    return


@app.cell(hide_code=True)
def _(debug_summary, mo, show_debug):
    if show_debug.value:
        mo.vstack([mo.md("### Debug summary"), mo.json(debug_summary)])
    return


if __name__ == "__main__":
    app.run()
