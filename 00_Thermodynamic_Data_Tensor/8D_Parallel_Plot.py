import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import sample_colorscale
from streamlit_plotly_events import plotly_events

# ============================================================
# Figure export utility
# ============================================================
def make_transparent_png_bytes(fig, width=1800, height=1200, scale=3):
    """
    Export a Plotly figure as a transparent-background PNG.
    Requires kaleido:
        pip install -U kaleido
    """
    fig_png = go.Figure(fig)

    fig_png.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig_png.to_image(
        format="png",
        width=width,
        height=height,
        scale=scale
    )



# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Interactive 8D Gibbs Parallel Plot",
    layout="wide"
)

st.title("Interactive 8D Parallel Plot for CoCrFeNi Gibbs Free Energy")


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("Data Settings")

    csv_folder = st.text_input("CSV folder", "csv_files")

    T_min_input = st.number_input("Minimum temperature [K]", value=300, step=100)
    T_max_input = st.number_input("Maximum temperature [K]", value=3300, step=100)
    T_step_input = st.number_input("Temperature step [K]", value=100, step=100)

    max_rows = st.slider(
        "Maximum plotted lines",
        min_value=50,
        max_value=3000,
        value=300,
        step=50
    )

    random_seed = st.number_input("Random seed", value=42, step=1)

    line_width = st.slider("Line width", 0.2, 3.0, 0.8, 0.1)
    line_opacity = st.slider("Line opacity", 0.05, 1.0, 0.85, 0.05)

    st.header("Display")
    show_raw_data = st.checkbox("Show loaded dataframe", value=False)
    show_normalized_data = st.checkbox("Show normalized dataframe", value=False)


# ============================================================
# Load data
# ============================================================
@st.cache_data
def load_gibbs_data(csv_folder, T_min_input, T_max_input, T_step_input):
    temperatures = list(range(T_min_input, T_max_input + 1, T_step_input))
    df_list = []

    missing_files = []

    for T in temperatures:
        file_path = os.path.join(csv_folder, f"Gibbs_{T}K.csv")

        if not os.path.exists(file_path):
            missing_files.append(file_path)
            continue

        df_T = pd.read_csv(file_path)
        df_T["T"] = T
        df_T["DeltaG"] = df_T["G_FCC"] - df_T["G_LIQ"]

        df_list.append(df_T)

    if len(df_list) == 0:
        return None, missing_files

    df_plot = pd.concat(df_list, ignore_index=True)
    return df_plot, missing_files


df_plot, missing_files = load_gibbs_data(
    csv_folder,
    int(T_min_input),
    int(T_max_input),
    int(T_step_input)
)

if df_plot is None:
    st.error("No valid CSV files were found. Check the folder path and filename pattern: Gibbs_<T>K.csv")
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

required_columns = set(variables)
missing_columns = required_columns - set(df_plot.columns)

if missing_columns:
    st.error(f"The following required columns are missing: {missing_columns}")
    st.stop()


# ============================================================
# Optional downsampling
# ============================================================
if len(df_plot) > max_rows:
    df_parallel = df_plot.sample(max_rows, random_state=int(random_seed)).copy()
else:
    df_parallel = df_plot.copy()

df_parallel = df_parallel.reset_index(drop=True)
df_parallel["Line_ID"] = df_parallel.index


# ============================================================
# Normalization
# ============================================================
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


# ============================================================
# Build custom hover text
# ============================================================
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
        f"<b>DeltaG:</b> {row['DeltaG'] / 1000:.6f} kJ/mol"
    )


df_parallel["hover_text"] = df_parallel.apply(make_hover_text, axis=1)


# ============================================================
# Plot setup
# ============================================================
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

x_vals = [x_positions[v] for v in variables]

T_min = df_parallel["T"].min()
T_max = df_parallel["T"].max()

fig = go.Figure()


# ============================================================
# Lines
# ============================================================
for idx, row in df_norm.iterrows():
    T_value = df_parallel.loc[idx, "T"]

    if np.isclose(T_max, T_min):
        T_norm = 0.5
    else:
        T_norm = (T_value - T_min) / (T_max - T_min)

    # PiYG reversed, similar to matplotlib PiYG_r
    color = sample_colorscale("PiYG", 1.0 - T_norm)[0]

    y_vals = [row[v] for v in variables]

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(
                color=color,
                width=line_width
            ),
            opacity=line_opacity,
            hoverinfo="text",
            hovertext=df_parallel.loc[idx, "hover_text"],
            customdata=[df_parallel.loc[idx, "Line_ID"]] * len(x_vals),
            showlegend=False,
            name=f"Line {idx}"
        )
    )


# ============================================================
# Vertical axes
# ============================================================
for var in variables:
    x = x_positions[var]

    fig.add_trace(
        go.Scatter(
            x=[x, x],
            y=[0, 1],
            mode="lines",
            line=dict(color="black", width=4),
            hoverinfo="skip",
            showlegend=False
        )
    )

    fig.add_annotation(
        x=x,
        y=1.08,
        text=f"<b>{axis_labels[var]}</b>",
        showarrow=False,
        font=dict(size=20, color="black")
    )


# ============================================================
# Horizontal composition guide lines
# ============================================================
for y in [0, 0.25, 0.50, 0.75, 1.0]:
    fig.add_trace(
        go.Scatter(
            x=[-0.15, 3],
            y=[y, y],
            mode="lines",
            line=dict(color="black", width=1.2, dash="dash"),
            hoverinfo="skip",
            showlegend=False
        )
    )


# ============================================================
# Axis min-max labels
# ============================================================
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


# ============================================================
# Composition labels on left side
# ============================================================
for label, y in zip(["0.00", "0.10", "0.20", "0.30", "0.40"], [0, 0.25, 0.50, 0.75, 1.0]):
    fig.add_annotation(
        x=-0.40,
        y=y,
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(size=15)
    )


# ============================================================
# Colorbar using invisible marker trace
# ============================================================
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
                tickvals=[T_min, (T_min + T_max) / 3, 2 * (T_min + T_max) / 3, T_max],
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


# ============================================================
# Layout
# ============================================================
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


# ============================================================
# Display interactive plot and capture hover
# ============================================================
st.subheader("Interactive Parallel Plot")

# ============================================================
# Download transparent PNG
# ============================================================
try:
    png_bytes = make_transparent_png_bytes(
        fig,
        width=1800,
        height=1200,
        scale=1
    )

    st.download_button(
        label="Download 8D plot as transparent PNG",
        data=png_bytes,
        file_name="CoCrFeNi_8D_parallel_plot_transparent.png",
        mime="image/png",
        type="primary"
    )

except Exception:
    st.warning(
        "PNG export requires the `kaleido` package. "
        "Install it with: `pip install -U kaleido`"
    )

hovered_points = plotly_events(
    fig,
    hover_event=True,
    click_event=True,
    select_event=False,
    override_height=760,
    override_width="100%"
)


# ============================================================
# Display hovered or clicked row below
# ============================================================
st.subheader("Hovered / Selected Line Values")

if hovered_points:
    curve_number = hovered_points[0]["curveNumber"]

    # Only actual data-line traces correspond to dataframe rows.
    # Axis lines and colorbar traces are added after the data traces.
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

        st.dataframe(
            pd.DataFrame([selected_row[variables]]),
            width='stretch'
        )
    else:
        st.info("Hover over a colored thermodynamic line to display its values.")
else:
    st.info("Hover over a colored line to display its Co, Cr, Fe, Ni, T, G_LIQ, G_FCC, and DeltaG values here.")


# ============================================================
# Optional dataframe display
# ============================================================
if show_raw_data:
    st.subheader("Raw Loaded Data")
    st.dataframe(df_parallel[variables], width='stretch')

if show_normalized_data:
    st.subheader("Normalized Plot Data")
    st.dataframe(df_norm, width='stretch')