import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from streamlit_plotly_events import plotly_events
from scipy.interpolate import LinearNDInterpolator


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="CoCrFeNi Gibbs Energy Explorer",
    layout="wide"
)

st.title("Co-Cr-Fe-Ni Gibbs Free Energy Explorer")

st.markdown(
    """
    This app visualizes the Gibbs free-energy data for the Co-Cr-Fe-Ni system.

    It contains two interactive views:

    **1. 8D Parallel Plot**  
    Shows the original discrete thermodynamic states from CSV files.

    **2. Interpolated Gibbs Tensor Plot**  
    Reconstructs the Gibbs-energy hypersurface at a selected temperature using interpolation in composition space.
    The stable phase is determined by:

    $$
    G_{stable} = \\min(G_{LIQ}, G_{FCC})
    $$
    """
)


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("Data Settings")

    csv_folder = st.text_input("CSV folder", "csv_files")

    T_min_input = st.number_input("Minimum temperature [K]", value=300, step=100)
    T_max_input = st.number_input("Maximum temperature [K]", value=3300, step=100)
    T_step_input = st.number_input("Temperature step [K]", value=100, step=100)

    st.header("8D Parallel Plot Settings")

    max_rows = st.slider(
        "Maximum plotted lines",
        min_value=20,
        max_value=500,
        value=100,
        step=10
    )

    random_seed = st.number_input("Random seed", value=42, step=1)

    line_width = st.slider("Line width", 0.2, 3.0, 0.8, 0.1)
    line_opacity = st.slider("Line opacity", 0.05, 1.0, 0.85, 0.05)

    show_raw_data = st.checkbox("Show raw dataframe", value=False)
    show_normalized_data = st.checkbox("Show normalized dataframe", value=False)


# ============================================================
# Load data
# ============================================================
@st.cache_data
def load_gibbs_data(csv_folder, T_min_input, T_max_input, T_step_input):
    """
    Load Gibbs_<T>K.csv files from the selected folder.
    Expected columns:
        Co, Cr, Fe, Ni, G_LIQ, G_FCC
    """

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
        "FCC"
    )

    return df_plot, missing_files


df_plot, missing_files = load_gibbs_data(
    csv_folder,
    int(T_min_input),
    int(T_max_input),
    int(T_step_input)
)

if df_plot is None:
    st.error("No valid CSV files were found. Check folder path and filename pattern: Gibbs_<T>K.csv")
    st.stop()

if missing_files:
    with st.expander("Missing files"):
        for f in missing_files:
            st.write(f)

st.success(f"Loaded data shape: {df_plot.shape}")


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
    "DeltaG"
]

axis_labels = {
    "Co": "Co",
    "Cr": "Cr",
    "Fe": "Fe",
    "Ni": "Ni",
    "T": "T [K]",
    "G_LIQ": "G_LIQ [kJ/mol]",
    "G_FCC": "G_FCC [kJ/mol]",
    "DeltaG": "DeltaG [kJ/mol]"
}


# ============================================================
# Interpolation function
# ============================================================
@st.cache_data(ttl=3600)
def build_interpolated_gibbs_grid(df, T_val, grid_res):
    """
    Build interpolated Gibbs-energy grid at one temperature.

    Independent composition coordinates:
        Co, Cr, Fe

    Dependent composition:
        Ni = 1 - Co - Cr - Fe

    Valid quaternary condition:
        Co + Cr + Fe + Ni = 1
        Co + Cr + Fe <= 1
    """

    df_T = df[df["T"] == T_val].copy()

    if df_T.empty:
        return None

    points = df_T[["Co", "Cr", "Fe"]].values

    interp_liq = LinearNDInterpolator(points, df_T["G_LIQ"].values)
    interp_fcc = LinearNDInterpolator(points, df_T["G_FCC"].values)

    x = np.linspace(0.0, 1.0, grid_res)

    Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")

    grid_points = np.column_stack([
        Xco.ravel(),
        Xcr.ravel(),
        Xfe.ravel()
    ])

    valid_mask = (
        grid_points[:, 0]
        + grid_points[:, 1]
        + grid_points[:, 2]
    ) <= 1.0

    pts_valid = grid_points[valid_mask]

    G_liq = interp_liq(pts_valid)
    G_fcc = interp_fcc(pts_valid)

    valid_eval = ~np.isnan(G_liq) & ~np.isnan(G_fcc)

    pts = pts_valid[valid_eval]
    G_liq = G_liq[valid_eval]
    G_fcc = G_fcc[valid_eval]

    if len(pts) == 0:
        return None

    Co = pts[:, 0]
    Cr = pts[:, 1]
    Fe = pts[:, 2]
    Ni = 1.0 - Co - Cr - Fe

    DeltaG = G_fcc - G_liq
    G_stable = np.minimum(G_liq, G_fcc)

    stable_phase = np.where(
        G_liq <= G_fcc,
        "LIQUID",
        "FCC"
    )

    df_interp = pd.DataFrame({
        "Co": Co,
        "Cr": Cr,
        "Fe": Fe,
        "Ni": Ni,
        "T": T_val,
        "G_LIQ": G_liq,
        "G_FCC": G_fcc,
        "DeltaG": DeltaG,
        "G_stable": G_stable,
        "Stable_Phase": stable_phase
    })

    return df_interp


# ============================================================
# Tabs
# ============================================================
tab1, tab2 = st.tabs([
    "8D Parallel Plot",
    "Interpolated Gibbs Energy Tensor"
])


# ============================================================
# TAB 1: 8D Parallel Plot
# ============================================================
with tab1:

    st.subheader("Interactive 8D Parallel Plot")

    if len(df_plot) > max_rows:
        df_parallel = df_plot.sample(max_rows, random_state=int(random_seed)).copy()
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
            "max": col_max
        }

        if np.isclose(col_max, col_min):
            df_norm[col] = 0.5
        else:
            df_norm[col] = (df_parallel[col] - col_min) / (col_max - col_min)

    def make_hover_text(row):
        return (
            f"<b>Line ID:</b> {int(row['Line_ID'])}<br>"
            f"<b>Co:</b> {row['Co']:.2f}<br>"
            f"<b>Cr:</b> {row['Cr']:.2f}<br>"
            f"<b>Fe:</b> {row['Fe']:.2f}<br>"
            f"<b>Ni:</b> {row['Ni']:.2f}<br>"
            f"<b>T:</b> {row['T']:.0f} K<br>"
            f"<b>G_LIQ:</b> {row['G_LIQ'] / 1000:.2f} kJ/mol<br>"
            f"<b>G_FCC:</b> {row['G_FCC'] / 1000:.2f} kJ/mol<br>"
            f"<b>DeltaG:</b> {row['DeltaG'] / 1000:.2f} kJ/mol<br>"
            f"<b>G_stable:</b> {row['G_stable'] / 1000:.2f} kJ/mol<br>"
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
        "DeltaG": 10
    }

    x_vals = np.array([x_positions[v] for v in variables])

    T_min = df_parallel["T"].min()
    T_max = df_parallel["T"].max()

    fig = go.Figure()

    # ========================================================
    # Colored thermodynamic strands only
    # These are the only hoverable/clickable objects.
    # ========================================================
    for idx, row in df_norm.iterrows():

        T_value = df_parallel.loc[idx, "T"]

        if np.isclose(T_max, T_min):
            T_norm = 0.5
        else:
            T_norm = (T_value - T_min) / (T_max - T_min)

        color = sample_colorscale("PiYG", 1.0 - T_norm)[0]

        y_vals = np.array([row[v] for v in variables])

        # ----------------------------------------------------
        # Add extra invisible hover points along each segment.
        # Plotly hover is easier near points than only lines.
        # ----------------------------------------------------
        x_hover = []
        y_hover = []

        n_hover_points_per_segment = 12

        for j in range(len(x_vals) - 1):
            xs = np.linspace(x_vals[j], x_vals[j + 1], n_hover_points_per_segment)
            ys = np.linspace(y_vals[j], y_vals[j + 1], n_hover_points_per_segment)

            x_hover.extend(xs)
            y_hover.extend(ys)

        fig.add_trace(
            go.Scatter(
                x=x_hover,
                y=y_hover,
                mode="lines+markers",
                line=dict(
                    color=color,
                    width=line_width
                ),
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.01
                ),
                opacity=line_opacity,
                hovertemplate=df_parallel.loc[idx, "hover_text"] + "<extra></extra>",
                customdata=[idx] * len(x_hover),
                showlegend=False,
                name=f"Line {idx}"
            )
        )

    # ========================================================
    # Vertical axes as shapes, not traces
    # These will NOT be hoverable.
    # ========================================================
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
                width=4
            )
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
            font=dict(size=20, color=label_color)
        )

    # ========================================================
    # Horizontal composition guide lines as shapes
    # These also will NOT be hoverable.
    # ========================================================
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
                dash="dash"
            )
        )

    # ========================================================
    # Axis min/max labels
    # ========================================================
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
            font=dict(size=13)
        )

        fig.add_annotation(
            x=x,
            y=1.035,
            text=f"<b>{max_text}</b>",
            showarrow=False,
            font=dict(size=13)
        )

    # ========================================================
    # Composition reference labels
    # ========================================================
    for label, y in zip(
        ["0.00", "0.10", "0.20", "0.30", "0.40"],
        [0, 0.25, 0.50, 0.75, 1.0]
    ):
        fig.add_annotation(
            x=-0.40,
            y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=15)
        )

    # ========================================================
    # Temperature colorbar
    # This is the only non-line trace, but it has no hover.
    # ========================================================
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
                        T_max
                    ],
                    ticktext=[
                        f"{T_min:.0f}",
                        f"{(T_min + T_max) / 3:.0f}",
                        f"{2 * (T_min + T_max) / 3:.0f}",
                        f"{T_max:.0f}"
                    ],
                    thickness=25,
                    len=0.75
                )
            ),
            hoverinfo="skip",
            showlegend=False
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
            showticklabels=False
        ),
        yaxis=dict(
            range=[-0.08, 1.15],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        )
    )

    clicked_points = plotly_events(
        fig,
        hover_event=False,   # no Streamlit rerun on hover
        click_event=True,    # print values only on click
        select_event=False,
        override_height=760,
        override_width="100%"
    )

    # st.subheader("Clicked Strand Values")

    # if clicked_points:
    #     curve_number = clicked_points[0]["curveNumber"]

    #     if curve_number < len(df_parallel):

    #         selected_row = df_parallel.iloc[curve_number]

    #         col1, col2, col3, col4 = st.columns(4)

    #         col1.metric("Co", f"{selected_row['Co']:.2f}")
    #         col2.metric("Cr", f"{selected_row['Cr']:.2f}")
    #         col3.metric("Fe", f"{selected_row['Fe']:.2f}")
    #         col4.metric("Ni", f"{selected_row['Ni']:.2f}")

    #         col5, col6, col7, col8 = st.columns(4)

    #         col5.metric("T", f"{selected_row['T']:.0f} K")
    #         col6.metric("G_LIQ", f"{selected_row['G_LIQ'] / 1000:.2f} kJ/mol")
    #         col7.metric("G_FCC", f"{selected_row['G_FCC'] / 1000:.2f} kJ/mol")
    #         col8.metric("DeltaG", f"{selected_row['DeltaG'] / 1000:.2f} kJ/mol")

    #         col9, col10 = st.columns(2)

    #         col9.metric("Stable G", f"{selected_row['G_stable'] / 1000:.2f} kJ/mol")
    #         col10.metric("Stable Phase", selected_row["Stable_Phase"])

    #         st.dataframe(
    #             pd.DataFrame([selected_row[
    #                 [
    #                     "Co",
    #                     "Cr",
    #                     "Fe",
    #                     "Ni",
    #                     "T",
    #                     "G_LIQ",
    #                     "G_FCC",
    #                     "DeltaG",
    #                     "G_stable",
    #                     "Stable_Phase"
    #                 ]
    #             ]]),
    #             width='stretch'
    #         )

    #     else:
    #         st.info("Click a colored thermodynamic strand to display values.")

    # else:
    #     st.info("Hover shows the thermodynamic state near the mouse. Click a thin colored strand to print its values here.")

    if show_raw_data:
        st.subheader("Raw Loaded Data")
        st.dataframe(df_parallel, width='stretch')

    if show_normalized_data:
        st.subheader("Normalized Plot Data")
        st.dataframe(df_norm, width='stretch')

# ============================================================
# TAB 2: Interpolated Gibbs Tensor
# ============================================================
with tab2:

    st.subheader("Interpolated Gibbs-Energy Tensor in Co-Cr-Fe-Ni Composition Space")

    col_a, col_b, col_c, col_d = st.columns(4)

    T_list = sorted(df_plot["T"].unique())

    with col_a:
        default_T = T_list[len(T_list) // 2]

        T_val = st.select_slider(
            "Temperature [K]",
            options=T_list,
            value=default_T
        )

    with col_b:
        grid_res = st.slider(
            "Composition grid resolution",
            min_value=10,
            max_value=45,
            value=25,
            step=5,
            help="Higher value gives finer interpolation but slower rendering."
        )

    with col_c:
        mode = st.selectbox(
            "Visualization mode",
            [
                "Stable Phase",
                "G_LIQ",
                "G_FCC",
                "DeltaG",
                "Both Phases Overlay"
            ]
        )

    with col_d:
        marker_size = st.slider(
            "Marker size",
            min_value=1,
            max_value=8,
            value=3
        )

    opacity = st.slider(
        "Opacity",
        min_value=0.2,
        max_value=1.0,
        value=0.75,
        step=0.05,
        key="tensor_opacity"
    )

    df_interp = build_interpolated_gibbs_grid(
        df_plot,
        T_val,
        grid_res
    )

    if df_interp is None or df_interp.empty:
        st.error(
            f"""
            No interpolated data available for T = {T_val} K.

            This usually means that the selected composition grid lies outside the convex hull
            of your original CSV composition points.
            Try lowering or changing the grid resolution.
            """
        )
        st.stop()

    fig3d = go.Figure()

    hover_text = (
        "Co: %{customdata[0]:.2f}<br>"
        "Cr: %{customdata[1]:.2f}<br>"
        "Fe: %{customdata[2]:.2f}<br>"
        "Ni: %{customdata[3]:.2f}<br>"
        "T: %{customdata[4]:.0f} K<br>"
        "G_LIQ: %{customdata[5]:.2f} kJ/mol<br>"
        "G_FCC: %{customdata[6]:.2f} kJ/mol<br>"
        "DeltaG: %{customdata[7]:.2f} kJ/mol<br>"
        "G_stable: %{customdata[8]:.2f} kJ/mol<br>"
        "Stable phase: %{customdata[9]}<extra></extra>"
    )

    custom_data = np.column_stack([
        df_interp["Co"],
        df_interp["Cr"],
        df_interp["Fe"],
        df_interp["Ni"],
        df_interp["T"],
        df_interp["G_LIQ"] / 1000,
        df_interp["G_FCC"] / 1000,
        df_interp["DeltaG"] / 1000,
        df_interp["G_stable"] / 1000,
        df_interp["Stable_Phase"]
    ])

    if mode == "Stable Phase":

        phase_color = np.where(
            df_interp["Stable_Phase"] == "LIQUID",
            0,
            1
        )

        fig3d.add_trace(
            go.Scatter3d(
                x=df_interp["Co"],
                y=df_interp["Cr"],
                z=df_interp["Fe"],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=df_interp["G_stable"] / 1000,
                    colorscale="plasma",
                    showscale=True,
                    colorbar=dict(title="G stable<br>[kJ/mol]")
                ),
                customdata=custom_data,
                hovertemplate=hover_text,
                name="Stable phase",
                opacity=opacity
            )
        )

    elif mode == "G_LIQ":

        fig3d.add_trace(
            go.Scatter3d(
                x=df_interp["Co"],
                y=df_interp["Cr"],
                z=df_interp["Fe"],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=df_interp["G_LIQ"] / 1000,
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="G_LIQ<br>[kJ/mol]")
                ),
                customdata=custom_data,
                hovertemplate=hover_text,
                name="LIQUID",
                opacity=opacity
            )
        )

    elif mode == "G_FCC":

        fig3d.add_trace(
            go.Scatter3d(
                x=df_interp["Co"],
                y=df_interp["Cr"],
                z=df_interp["Fe"],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=df_interp["G_FCC"] / 1000,
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(title="G_FCC<br>[kJ/mol]")
                ),
                customdata=custom_data,
                hovertemplate=hover_text,
                name="FCC",
                opacity=opacity
            )
        )

    elif mode == "DeltaG":

        fig3d.add_trace(
            go.Scatter3d(
                x=df_interp["Co"],
                y=df_interp["Cr"],
                z=df_interp["Fe"],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=df_interp["DeltaG"] / 1000,
                    colorscale="RdBu",
                    showscale=True,
                    colorbar=dict(title="DeltaG<br>[kJ/mol]")
                ),
                customdata=custom_data,
                hovertemplate=hover_text,
                name="DeltaG",
                opacity=opacity
            )
        )

    else:

        fig3d.add_trace(
            go.Scatter3d(
                x=df_interp["Co"],
                y=df_interp["Cr"],
                z=df_interp["Fe"],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=df_interp["G_LIQ"] / 1000,
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="G_LIQ<br>[kJ/mol]", x=1.02)
                ),
                customdata=custom_data,
                hovertemplate=hover_text,
                name="LIQUID",
                opacity=opacity
            )
        )

        fig3d.add_trace(
            go.Scatter3d(
                x=df_interp["Co"],
                y=df_interp["Cr"],
                z=df_interp["Fe"],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=df_interp["G_FCC"] / 1000,
                    colorscale="Reds"
                ),
                customdata=custom_data,
                hovertemplate=hover_text,
                name="FCC",
                opacity=opacity
            )
        )

    fig3d.update_layout(
        title=f"Interpolated Gibbs Energy Tensor at T = {T_val} K | Points = {len(df_interp):,}",
        height=780,
        scene=dict(
            xaxis_title="x<sub>Co</sub>",
            yaxis_title="x<sub>Cr</sub>",
            zaxis_title="x<sub>Fe</sub>",
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    clicked_3d = plotly_events(
        fig3d,
        hover_event=False,  # no Streamlit rerun on hover
        click_event=True,   # print values only on click
        select_event=False,
        override_height=780,
        override_width="100%"
    )

    st.subheader("Clicked Interpolated State")

    if clicked_3d:
        point_id = clicked_3d[0]["pointNumber"]

        selected_row = df_interp.iloc[point_id]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Co", f"{selected_row['Co']:.2f}")
        c2.metric("Cr", f"{selected_row['Cr']:.2f}")
        c3.metric("Fe", f"{selected_row['Fe']:.2f}")
        c4.metric("Ni", f"{selected_row['Ni']:.2f}")

        c5, c6, c7, c8 = st.columns(4)

        c5.metric("T", f"{selected_row['T']:.0f} K")
        c6.metric("G_LIQ", f"{selected_row['G_LIQ'] / 1000:.2f} kJ/mol")
        c7.metric("G_FCC", f"{selected_row['G_FCC'] / 1000:.2f} kJ/mol")
        c8.metric("DeltaG", f"{selected_row['DeltaG'] / 1000:.2f} kJ/mol")

        c9, c10 = st.columns(2)

        c9.metric("Stable G", f"{selected_row['G_stable'] / 1000:.2f} kJ/mol")
        c10.metric("Stable Phase", selected_row["Stable_Phase"])

        st.dataframe(
            pd.DataFrame([selected_row]),
            width='content'
        )

    else:
        st.info("Hover shows the thermodynamic state near the mouse. Click an interpolated point to print its values here.")
        
    # st.subheader("Phase Statistics at Current Temperature")

    liq_fraction = np.sum(df_interp["Stable_Phase"] == "LIQUID") / len(df_interp) * 100
    fcc_fraction = np.sum(df_interp["Stable_Phase"] == "FCC") / len(df_interp) * 100

    # s1, s2, s3, s4 = st.columns(4)

    # s1.metric("Minimum stable G", f"{df_interp['G_stable'].min() / 1000:.2f} kJ/mol")
    # s2.metric("Maximum stable G", f"{df_interp['G_stable'].max() / 1000:.2f} kJ/mol")
    # s3.metric("LIQUID region", f"{liq_fraction:.2f}%")
    # s4.metric("FCC region", f"{fcc_fraction:.2f}%")

    with st.expander("Interpolated dataframe"):
        st.dataframe(df_interp, width='stretch')













































































































# import os
# import glob
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go
# from scipy.interpolate import LinearNDInterpolator

# # ================= CONFIGURATION =================
# st.set_page_config(page_title="CoCrFeNi Gibbs Energy Explorer", layout="wide")
# st.title("🔷 Co-Cr-Fe-Ni Gibbs Free Energy Tensor Visualization")
# st.markdown("""
# This app reconstructs the continuous $G(\mathbf{x}, T)$ hypersurface from discrete CSV data.  
# The stable phase is determined by $G_{\text{stable}} = \min(G_{\text{LIQ}}, G_{\text{FCC}})$.
# """)

# # ================= DATA LOADING =================
# @st.cache_data
# def load_all_data(csv_dir="csv_files"):
#     """Load and concatenate all Gibbs_*.csv files."""
#     files = glob.glob(os.path.join(csv_dir, "Gibbs_*.csv"))
#     if not files:
#         st.error("No CSV files found in 'csv_files/' directory.")
#         st.stop()
        
#     dfs = []
#     for f in files:
#         basename = os.path.basename(f)
#         try:
#             T = int(basename.replace("Gibbs_", "").replace("K.csv", ""))
#             df = pd.read_csv(f, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
#             df["T"] = T
#             dfs.append(df)
#         except Exception as e:
#             st.warning(f"Skipping {f}: {e}")
            
#     return pd.concat(dfs, ignore_index=True)

# df = load_all_data()

# # ================= INTERPOLATION =================
# @st.cache_data(ttl=3600)
# def build_interpolators_for_T(df, T):
#     """Build 3D interpolators for a specific temperature slice."""
#     df_T = df[df["T"] == T].copy()
#     if len(df_T) == 0:
#         return None, None
    
#     # Points in 3D composition space
#     pts = df_T[["Co", "Cr", "Fe"]].values
#     interp_liq = LinearNDInterpolator(pts, df_T["G_LIQ"].values)
#     interp_fcc = LinearNDInterpolator(pts, df_T["G_FCC"].values)
#     return interp_liq, interp_fcc

# # ================= UI CONTROLS =================
# with st.sidebar:
#     st.header("🎛️ Parameters")
    
#     T_list = sorted(df["T"].unique())
#     T_val = st.select_slider("Temperature (K)", options=T_list, value=1000)
    
#     grid_res = st.slider("Composition Grid Resolution", 15, 40, 25, step=5, help="Higher = finer detail but slower rendering.")
    
#     show_phase = st.radio("Visualization Mode", 
#                           ["Stable Phase (Min G)", "LIQUID Only", "FCC Only", "Both Phases Overlay"])
    
#     marker_size = st.slider("Marker Size", 1, 6, 3)
#     opacity = st.slider("Opacity", 0.3, 1.0, 0.75, 0.05)

# # ================= COMPUTATION & PLOTTING =================
# if T_val is not None:
#     interp_liq, interp_fcc = build_interpolators_for_T(df, T_val)
    
#     if interp_liq is None:
#         st.error(f"No data loaded for T = {T_val} K.")
#     else:
#         # Generate tetrahedral grid
#         x = np.linspace(0, 1, grid_res)
#         Xco, Xcr, Xfe = np.meshgrid(x, x, x, indexing="ij")
#         grid_pts = np.column_stack([Xco.ravel(), Xcr.ravel(), Xfe.ravel()])
        
#         # Mask: valid quaternary compositions must satisfy sum <= 1
#         valid_mask = (grid_pts[:, 0] + grid_pts[:, 1] + grid_pts[:, 2]) <= 1.0
#         pts_valid = grid_pts[valid_mask]
        
#         # Evaluate interpolators
#         G_liq = interp_liq(pts_valid)
#         G_fcc = interp_fcc(pts_valid)
        
#         # Handle extrapolation/NaNs (outside convex hull)
#         valid_eval = ~np.isnan(G_liq) & ~np.isnan(G_fcc)
#         pts = pts_valid[valid_eval]
#         G_liq = G_liq[valid_eval]
#         G_fcc = G_fcc[valid_eval]
        
#         # Determine stable phase & minimum G
#         G_stable = np.minimum(G_liq, G_fcc)
#         stable_label = np.where(G_liq <= G_fcc, "LIQUID", "FCC")
        
#         # Build Plotly Figure
#         fig = go.Figure()
        
#         if show_phase == "Stable Phase (Min G)":
#             fig.add_trace(go.Scatter3d(
#                 x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
#                 mode="markers",
#                 marker=dict(size=marker_size, color=G_stable, colorscale="Viridis", 
#                            showscale=True, colorbar=dict(title="G (J/mol)", thickness=20)),
#                 name="Stable Phase",
#                 opacity=opacity
#             ))
#         elif show_phase == "LIQUID Only":
#             fig.add_trace(go.Scatter3d(
#                 x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
#                 mode="markers",
#                 marker=dict(size=marker_size, color=G_liq, colorscale="Blues", 
#                            showscale=True, colorbar=dict(title="G_LIQ (J/mol)", thickness=20)),
#                 name="LIQUID", opacity=opacity
#             ))
#         elif show_phase == "FCC Only":
#             fig.add_trace(go.Scatter3d(
#                 x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
#                 mode="markers",
#                 marker=dict(size=marker_size, color=G_fcc, colorscale="Reds", 
#                            showscale=True, colorbar=dict(title="G_FCC (J/mol)", thickness=20)),
#                 name="FCC", opacity=opacity
#             ))
#         else:
#             fig.add_trace(go.Scatter3d(
#                 x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
#                 mode="markers", marker=dict(size=marker_size, color=G_liq, colorscale="Blues", opacity=opacity),
#                 name="LIQUID"
#             ))
#             fig.add_trace(go.Scatter3d(
#                 x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
#                 mode="markers", marker=dict(size=marker_size, color=G_fcc, colorscale="Reds", opacity=opacity),
#                 name="FCC"
#             ))
            
#         fig.update_layout(
#             scene=dict(
#                 xaxis_title="x<sub>Co</sub>", yaxis_title="x<sub>Cr</sub>", zaxis_title="x<sub>Fe</sub>",
#                 aspectmode="cube",
#                 camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
#             ),
#             title=f"Gibbs Energy Tensor at T = {T_val} K | Points: {len(pts):,}",
#             margin=dict(l=0, r=0, b=0, t=40)
#         )
        
#         st.plotly_chart(fig, width='content')
        
#         # Quick stats
#         st.subheader("📊 Phase Statistics at Current Grid")
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Min G (Stable)", f"{G_stable.min():,.0f} J/mol")
#         col2.metric("Max G", f"{G_stable.max():,.0f} J/mol")
#         if show_phase in ["Stable Phase (Min G)", "Both Phases Overlay"]:
#             liq_pct = np.sum(G_liq <= G_fcc) / len(G_liq) * 100
#             col3.metric("LIQUID Region", f"{liq_pct:.1f}%")
