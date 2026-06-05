# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.10.0",
#     "numpy",
#     "pandas",
#     "plotly",
#     "kaleido",
# ]
# [tool.marimo.runtime]
# auto_instantiate = true
# on_cell_change = "autorun"
# watcher_on_save = "autorun"
# [tool.marimo.display]
# default_width = "full"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import io
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    import marimo as mo

    return go, mo, np, os, pd, sample_colorscale


@app.cell
def _(mo):
    mo.md(r"""
    # Interactive 8D Parallel Plot for CoCrFeNi Gibbs Free Energy

    This marimo notebook is a webapp-style version of the original Streamlit script.
    It automatically loads `csv_files/Gibbs_<T>K.csv`, builds the 8D parallel plot, and updates when you change the controls.
    """)
    return


@app.cell
def _(mo):
    csv_folder = mo.ui.text(value="csv_files", label="CSV folder")
    T_min_input = mo.ui.number(start=0, stop=10000, step=100, value=300, label="Minimum temperature [K]")
    T_max_input = mo.ui.number(start=0, stop=10000, step=100, value=3300, label="Maximum temperature [K]")
    T_step_input = mo.ui.number(start=1, stop=5000, step=100, value=100, label="Temperature step [K]")

    max_rows = mo.ui.slider(start=50, stop=3000, step=50, value=300, label="Maximum plotted lines")
    random_seed = mo.ui.number(start=0, stop=999999, step=1, value=42, label="Random seed")
    line_width = mo.ui.slider(start=0.2, stop=3.0, step=0.1, value=0.8, label="Line width")
    line_opacity = mo.ui.slider(start=0.05, stop=1.0, step=0.05, value=0.85, label="Line opacity")

    show_raw_data = mo.ui.checkbox(value=False, label="Show loaded dataframe")
    show_normalized_data = mo.ui.checkbox(value=False, label="Show normalized dataframe")

    controls = mo.vstack(
        [
            mo.md("### Data Settings"),
            csv_folder,
            mo.hstack([T_min_input, T_max_input, T_step_input], justify="start"),
            mo.hstack([max_rows, random_seed], justify="start"),
            mo.hstack([line_width, line_opacity], justify="start"),
            mo.md("### Display"),
            mo.hstack([show_raw_data, show_normalized_data], justify="start"),
        ],
        gap=1,
    )
    controls
    return (
        T_max_input,
        T_min_input,
        T_step_input,
        csv_folder,
        line_opacity,
        line_width,
        max_rows,
        random_seed,
        show_normalized_data,
        show_raw_data,
    )


@app.cell
def _(T_max_input, T_min_input, T_step_input, csv_folder, os, pd):
    def load_gibbs_data(csv_folder_value, T_min_value, T_max_value, T_step_value):
        temperatures = list(range(int(T_min_value), int(T_max_value) + 1, int(T_step_value)))
        df_list = []
        missing_files = []
        read_errors = []

        required_cols = ["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"]

        for T in temperatures:
            file_path = os.path.join(str(csv_folder_value), f"Gibbs_{T}K.csv")
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                continue

            try:
                df_T = pd.read_csv(file_path)
                missing_cols = [c for c in required_cols if c not in df_T.columns]
                if missing_cols:
                    read_errors.append(f"{file_path}: missing columns {missing_cols}")
                    continue

                df_T = df_T.copy()
                df_T["T"] = T
                df_T["DeltaG"] = df_T["G_FCC"] - df_T["G_LIQ"]
                df_list.append(df_T)
            except Exception as exc:
                read_errors.append(f"{file_path}: {exc}")

        if len(df_list) == 0:
            return None, missing_files, read_errors

        df_plot = pd.concat(df_list, ignore_index=True)
        return df_plot, missing_files, read_errors

    df_plot, missing_files, read_errors = load_gibbs_data(
        csv_folder.value,
        T_min_input.value,
        T_max_input.value,
        T_step_input.value,
    )
    return df_plot, missing_files, read_errors


@app.cell
def _(df_plot, missing_files, mo, read_errors):
    if df_plot is None:
        load_status = mo.vstack(
            [
                mo.callout(
                    "No valid CSV files were found. Check the folder path and filename pattern: `Gibbs_<T>K.csv`.",
                    kind="danger",
                ),
                mo.accordion({"Missing files": "\n".join(missing_files[:200]) if missing_files else "None"}),
            ]
        )
    else:
        status_items = [mo.callout(f"Loaded data shape: `{df_plot.shape}`", kind="success")]
        if missing_files:
            status_items.append(
                mo.accordion({"Missing files": "\n".join(missing_files[:300])})
            )
        if read_errors:
            status_items.append(
                mo.accordion({"Read errors": "\n".join(read_errors[:100])})
            )
        load_status = mo.vstack(status_items)
    load_status
    return


@app.cell
def _(df_plot, max_rows, np, pd, random_seed):
    variables = ["Co", "Cr", "Fe", "Ni", "T", "G_LIQ", "G_FCC", "DeltaG"]

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

    if df_plot is None:
        df_parallel = pd.DataFrame(columns=variables + ["Line_ID", "hover_text"])
        df_norm = pd.DataFrame(columns=variables)
        axis_limits = {}
        missing_columns = set()
    else:
        required_columns = set(variables)
        missing_columns = required_columns - set(df_plot.columns)

        if missing_columns:
            df_parallel = pd.DataFrame(columns=variables + ["Line_ID", "hover_text"])
            df_norm = pd.DataFrame(columns=variables)
            axis_limits = {}
        else:
            if len(df_plot) > int(max_rows.value):
                df_parallel = df_plot.sample(
                    int(max_rows.value),
                    random_state=int(random_seed.value),
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
                axis_limits[col] = {"min": col_min, "max": col_max}

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
                    f"<b>DeltaG:</b> {row['DeltaG'] / 1000:.6f} kJ/mol"
                )

            df_parallel["hover_text"] = df_parallel.apply(make_hover_text, axis=1)
    return (
        axis_labels,
        axis_limits,
        df_norm,
        df_parallel,
        missing_columns,
        variables,
    )


@app.cell
def _(
    axis_labels,
    axis_limits,
    df_norm,
    df_parallel,
    go,
    line_opacity,
    line_width,
    missing_columns,
    mo,
    np,
    sample_colorscale,
    variables,
):
    def make_transparent_png_bytes(fig, width=1800, height=1200, scale=1):
        fig_png = go.Figure(fig)
        fig_png.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig_png.to_image(format="png", width=width, height=height, scale=scale)

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
    x_vals = [x_positions[v] for v in variables]

    fig = go.Figure()
    plot_error = None

    if missing_columns:
        plot_error = f"The following required columns are missing: {sorted(missing_columns)}"
    elif len(df_parallel) == 0:
        plot_error = "No data available to plot."
    else:
        T_min = df_parallel["T"].min()
        T_max = df_parallel["T"].max()

        for idx, row in df_norm.iterrows():
            T_value = df_parallel.loc[idx, "T"]
            if np.isclose(T_max, T_min):
                T_norm = 0.5
            else:
                T_norm = (T_value - T_min) / (T_max - T_min)

            color = sample_colorscale("PiYG", 1.0 - T_norm)[0]
            y_vals = [row[v] for v in variables]

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    line=dict(color=color, width=float(line_width.value)),
                    marker=dict(size=5, color=color, opacity=0.01),
                    opacity=float(line_opacity.value),
                    hoverinfo="text",
                    hovertext=df_parallel.loc[idx, "hover_text"],
                    customdata=[int(df_parallel.loc[idx, "Line_ID"])] * len(x_vals),
                    showlegend=False,
                    name=f"Line {idx}"
                )
            )

        for var in variables:
            x = x_positions[var]
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color="black", width=4),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_annotation(
                x=x,
                y=1.08,
                text=f"<b>{axis_labels[var]}</b>",
                showarrow=False,
                font=dict(size=20, color="black"),
            )

        for y in [0, 0.25, 0.50, 0.75, 1.0]:
            fig.add_trace(
                go.Scatter(
                    x=[-0.15, 3],
                    y=[y, y],
                    mode="lines",
                    line=dict(color="black", width=1.2, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
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

        for label, y in zip(["0.00", "0.10", "0.20", "0.30", "0.40"], [0, 0.25, 0.50, 0.75, 1.0]):
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
                        tickvals=[T_min, (T_min + T_max) / 3, 2 * (T_min + T_max) / 3, T_max],
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
            dragmode="select",
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

    if plot_error:
        plot_ui = mo.callout(plot_error, kind="danger")
    else:
        plot_ui = mo.ui.plotly(
            fig,
            config={
                "displaylogo": False,
                "responsive": True,
                "modeBarButtonsToAdd": ["select2d", "lasso2d"],
            },
            label="Interactive Parallel Plot",
        )
    return fig, make_transparent_png_bytes, plot_error, plot_ui


@app.cell
def _(fig, make_transparent_png_bytes, mo, plot_error, plot_ui):
    if plot_error:
        png_download = mo.callout(
            "PNG export is disabled because the plot was not created.",
            kind="warn",
        )
    else:
        def _png_data():
            try:
                return make_transparent_png_bytes(fig, width=1800, height=1200, scale=1)
            except Exception as exc:
                return f"PNG export failed. Install/upgrade kaleido. Error: {exc}".encode("utf-8")

        png_download = mo.download(
            data=_png_data,
            filename="CoCrFeNi_8D_parallel_plot_transparent.png",
            mimetype="image/png",
            label="Download 8D plot as transparent PNG",
        )

    mo.vstack(
        [
            mo.md("## Interactive Parallel Plot"),
            png_download,
            plot_ui,
        ],
        gap=1,
    )
    return


@app.cell
def _(df_parallel, mo, pd, plot_ui, variables):
    selected_row = None

    try:
        selected_points = plot_ui.value if hasattr(plot_ui, "value") else []
    except Exception:
        selected_points = []

    if selected_points and len(df_parallel) > 0:
        point = selected_points[0]
        curve_number = point.get("curveNumber", point.get("curve_number", None))
        customdata = point.get("customdata", None)

        if customdata is not None:
            if isinstance(customdata, (list, tuple)) and len(customdata) > 0:
                line_id = int(customdata[0])
            else:
                line_id = int(customdata)
        elif curve_number is not None:
            line_id = int(curve_number)
        else:
            line_id = None

        if line_id is not None and 0 <= line_id < len(df_parallel):
            selected_row = df_parallel.iloc[line_id]

    if selected_row is None:
        selected_output = mo.callout(
            "Hover over a colored line to see its values in the Plotly tooltip. Drag-select or click a line/marker to print its Co, Cr, Fe, Ni, T, G_LIQ, G_FCC, and DeltaG values here.",
            kind="info",
        )
    else:
        selected_df = pd.DataFrame([selected_row[variables]])
        metrics_md = mo.md(
            f"""
            | Co | Cr | Fe | Ni |
            |---:|---:|---:|---:|
            | {selected_row['Co']:.6f} | {selected_row['Cr']:.6f} | {selected_row['Fe']:.6f} | {selected_row['Ni']:.6f} |

            | T | G_LIQ | G_FCC | DeltaG |
            |---:|---:|---:|---:|
            | {selected_row['T']:.0f} K | {selected_row['G_LIQ'] / 1000:.6f} kJ/mol | {selected_row['G_FCC'] / 1000:.6f} kJ/mol | {selected_row['DeltaG'] / 1000:.6f} kJ/mol |
            """
        )
        selected_output = mo.vstack([metrics_md, mo.ui.table(selected_df, pagination=False)])

    mo.vstack(
        [
            mo.md("## Hovered / Selected Line Values"),
            selected_output,
        ],
        gap=1,
    )
    return


@app.cell
def _(
    df_norm,
    df_parallel,
    mo,
    show_normalized_data,
    show_raw_data,
    variables,
):
    optional_outputs = []
    if show_raw_data.value and len(df_parallel) > 0:
        optional_outputs.append(mo.md("## Raw Loaded Data"))
        optional_outputs.append(mo.ui.table(df_parallel[variables], pagination=True))

    if show_normalized_data.value and len(df_norm) > 0:
        optional_outputs.append(mo.md("## Normalized Plot Data"))
        optional_outputs.append(mo.ui.table(df_norm, pagination=True))

    if optional_outputs:
        mo.vstack(optional_outputs, gap=1)
    return


if __name__ == "__main__":
    app.run()
