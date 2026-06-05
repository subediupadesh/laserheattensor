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
app = marimo.App(width="full", app_title="Interactive 8D Gibbs Parallel Plot")


@app.cell(hide_code=True)
def _():
    import os
    import io
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    import marimo as mo
    return go, io, mo, np, os, pd, sample_colorscale


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Interactive 8D Parallel Plot for CoCrFeNi Gibbs Free Energy

        This is a marimo app version of the original Streamlit dashboard. It loads
        `Gibbs_<T>K.csv` files from the `csv_files` folder and creates the same 8D
        parallel-coordinate style visualization for Co, Cr, Fe, Ni, temperature,
        `G_LIQ`, `G_FCC`, and `DeltaG`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # csv_folder = mo.ui.text(value="../0_Thermodynamic_Data_Tensor/csv_files/", label="CSV folder") ## For local use
    csv_folder = mo.ui.text(value="0_Thermodynamic_Data_Tensor/csv_files/", label="CSV folder")
    T_min_input = mo.ui.number(value=300, step=100, label="Minimum temperature [K]")
    T_max_input = mo.ui.number(value=3300, step=100, label="Maximum temperature [K]")
    T_step_input = mo.ui.number(value=100, step=100, label="Temperature step [K]")

    max_rows = mo.ui.slider(start=50, stop=3000, step=50, value=300, label="Maximum plotted lines")
    random_seed = mo.ui.number(value=42, step=1, label="Random seed")
    line_width = mo.ui.slider(start=0.2, stop=3.0, step=0.1, value=0.8, label="Line width")
    line_opacity = mo.ui.slider(start=0.05, stop=1.0, step=0.05, value=0.85, label="Line opacity")

    show_raw_data = mo.ui.checkbox(value=False, label="Show loaded dataframe")
    show_normalized_data = mo.ui.checkbox(value=False, label="Show normalized dataframe")

    sidebar = mo.sidebar(
        mo.vstack(
            [
                mo.md("## Data Settings"),
                csv_folder,
                T_min_input,
                T_max_input,
                T_step_input,
                mo.md("## Plot Settings"),
                max_rows,
                random_seed,
                line_width,
                line_opacity,
                mo.md("## Display"),
                show_raw_data,
                show_normalized_data,
            ],
            gap=1,
        )
    )
    sidebar
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


@app.cell(hide_code=True)
def _(np, os, pd):
    def load_gibbs_data(csv_folder_value, T_min_value, T_max_value, T_step_value):
        temperatures = list(range(int(T_min_value), int(T_max_value) + 1, int(T_step_value)))
        df_list = []
        missing_files = []

        for T in temperatures:
            file_path = os.path.join(str(csv_folder_value), f"Gibbs_{T}K.csv")
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                continue

            try:
                df_T = pd.read_csv(file_path)
            except Exception as exc:
                missing_files.append(f"{file_path}  [read error: {exc}]")
                continue

            df_T["T"] = T
            if "G_FCC" in df_T.columns and "G_LIQ" in df_T.columns:
                df_T["DeltaG"] = df_T["G_FCC"] - df_T["G_LIQ"]
            df_list.append(df_T)

        if len(df_list) == 0:
            return None, missing_files

        df_plot = pd.concat(df_list, ignore_index=True)
        return df_plot, missing_files


    def make_transparent_png_bytes(fig, width=1800, height=1200, scale=1):
        fig_png = fig.__class__(fig)
        fig_png.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig_png.to_image(format="png", width=width, height=height, scale=scale)


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


    def safe_numeric(value, fallback):
        try:
            if value is None:
                return fallback
            if isinstance(value, str) and value.strip() == "":
                return fallback
            return value
        except Exception:
            return fallback

    return load_gibbs_data, make_hover_text, make_transparent_png_bytes, safe_numeric


@app.cell(hide_code=True)
def _(
    T_max_input,
    T_min_input,
    T_step_input,
    csv_folder,
    load_gibbs_data,
    mo,
    safe_numeric,
):
    T_min_value = int(safe_numeric(T_min_input.value, 300))
    T_max_value = int(safe_numeric(T_max_input.value, 3300))
    T_step_value = int(safe_numeric(T_step_input.value, 100))

    if T_step_value <= 0:
        T_step_value = 100

    df_plot, missing_files = load_gibbs_data(
        csv_folder.value,
        T_min_value,
        T_max_value,
        T_step_value,
    )

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
        data_status = mo.callout(
            f"No valid CSV files were found. Check folder path and filename pattern: `Gibbs_<T>K.csv`. Current folder: `{csv_folder.value}`",
            kind="danger",
        )
    else:
        data_status = mo.callout(f"Loaded data shape: `{df_plot.shape}`", kind="success")

    missing_status = None
    if missing_files:
        preview = "\n".join([f"- `{item}`" for item in missing_files[:40]])
        extra = "" if len(missing_files) <= 40 else f"\n- ... and {len(missing_files) - 40} more"
        missing_status = mo.accordion({"Missing files": mo.md(preview + extra)})

    return axis_labels, data_status, df_plot, missing_files, missing_status, variables


@app.cell(hide_code=True)
def _(
    df_plot,
    line_opacity,
    line_width,
    make_hover_text,
    max_rows,
    mo,
    np,
    pd,
    random_seed,
    sample_colorscale,
    safe_numeric,
    variables,
):
    required_columns = set(variables)

    if df_plot is None:
        df_parallel = pd.DataFrame(columns=variables + ["Line_ID", "hover_text"])
        df_norm = pd.DataFrame(columns=variables)
        axis_limits = {}
        processing_status = mo.md("")
    else:
        missing_columns = required_columns - set(df_plot.columns)
        if missing_columns:
            df_parallel = pd.DataFrame(columns=variables + ["Line_ID", "hover_text"])
            df_norm = pd.DataFrame(columns=variables)
            axis_limits = {}
            processing_status = mo.callout(
                f"The following required columns are missing: `{sorted(missing_columns)}`",
                kind="danger",
            )
        else:
            max_rows_value = int(safe_numeric(max_rows.value, 300))
            random_seed_value = int(safe_numeric(random_seed.value, 42))
            if len(df_plot) > max_rows_value:
                df_parallel = df_plot.sample(max_rows_value, random_state=random_seed_value).copy()
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

            df_parallel["hover_text"] = df_parallel.apply(make_hover_text, axis=1)
            processing_status = mo.md("")

    return axis_limits, df_norm, df_parallel, processing_status


@app.cell(hide_code=True)
def _(
    axis_labels,
    axis_limits,
    df_norm,
    df_parallel,
    go,
    line_opacity,
    line_width,
    np,
    sample_colorscale,
    safe_numeric,
    variables,
):
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

    if len(df_parallel) > 0:
        T_min = df_parallel["T"].min()
        T_max = df_parallel["T"].max()
        lw = float(safe_numeric(line_width.value, 0.8))
        lo = float(safe_numeric(line_opacity.value, 0.85))

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
                    line=dict(color=color, width=lw),
                    marker=dict(size=4, opacity=0.001),
                    opacity=lo,
                    hoverinfo="text",
                    hovertext=df_parallel.loc[idx, "hover_text"],
                    customdata=[int(df_parallel.loc[idx, "Line_ID"])] * len(x_vals),
                    showlegend=False,
                    name=f"Line {idx}",
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

            fig.add_annotation(x=x, y=-0.055, text=f"<b>{min_text}</b>", showarrow=False, font=dict(size=13))
            fig.add_annotation(x=x, y=1.035, text=f"<b>{max_text}</b>", showarrow=False, font=dict(size=13))

        for label, y in zip(["0.00", "0.10", "0.20", "0.30", "0.40"], [0, 0.25, 0.50, 0.75, 1.0]):
            fig.add_annotation(x=-0.40, y=y, text=f"<b>{label}</b>", showarrow=False, font=dict(size=15))

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
                        ticktext=[f"{T_min:.0f}", f"{(T_min + T_max) / 3:.0f}", f"{2 * (T_min + T_max) / 3:.0f}", f"{T_max:.0f}"],
                        thickness=25,
                        len=0.75,
                    ),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # fig.update_layout(
    #     height=760,
    #     margin=dict(l=40, r=80, t=80, b=80),
    #     plot_bgcolor="white",
    #     paper_bgcolor="white",
    #     hovermode="closest",
    #     dragmode="select",
    #     xaxis=dict(range=[-0.7, 10.6], showgrid=False, zeroline=False, showticklabels=False),
    #     yaxis=dict(range=[-0.08, 1.15], showgrid=False, zeroline=False, showticklabels=False),
    # )
    fig.update_layout(
        height=760,
        margin=dict(l=40, r=80, t=80, b=80),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        dragmode="select",
        xaxis=dict(range=[-0.7, 10.6], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.08, 1.15], showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig,


@app.cell(hide_code=True)
def _(df_parallel, mo):
    if len(df_parallel) > 0:
        line_id_options = ["None"] + [str(i) for i in df_parallel["Line_ID"].tolist()]
    else:
        line_id_options = ["None"]

    manual_line_id = mo.ui.dropdown(
        options=line_id_options,
        value="None",
        label="Manual Line_ID selection",
    )
    return manual_line_id,


@app.cell(hide_code=True)
def _(fig, mo):
    interactive_plot = mo.ui.plotly(
        fig,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "CoCrFeNi_8D_parallel_plot_transparent",
                "height": 1200,
                "width": 1800,
                "scale": 1,
            },
        },
    )
    return interactive_plot,


# @app.cell(hide_code=True)
# def _(fig, make_transparent_png_bytes, mo):
#     try:
#         png_bytes = make_transparent_png_bytes(fig, width=1800, height=1200, scale=1)
#         png_download = mo.download(
#             data=png_bytes,
#             filename="CoCrFeNi_8D_parallel_plot_transparent.png",
#             mimetype="image/png",
#             label="Download 8D plot as transparent PNG",
#         )
#     except Exception:
#         png_download = mo.callout(
#             "PNG export requires a compatible `kaleido` installation. The interactive Plotly toolbar can still export a normal PNG.",
#             kind="warn",
#         )
#     return png_download,

@app.cell(hide_code=True)
def _(mo):
    png_download = mo.callout(
        "Transparent PNG export is disabled in the GitHub Pages/WASM version. Use the Plotly toolbar camera button to export the figure.",
        kind="info",
    )
    return png_download,

@app.cell(hide_code=True)
def _(df_parallel, interactive_plot, manual_line_id, mo, pd, variables):
    selected_line_id = None

    if manual_line_id.value != "None":
        selected_line_id = int(manual_line_id.value)
    else:
        selection_value = interactive_plot.value
        if selection_value:
            first_point = selection_value[0]
            custom_data = first_point.get("customdata", None)
            if isinstance(custom_data, list) and len(custom_data) > 0:
                selected_line_id = int(custom_data[0])
            elif custom_data is not None:
                selected_line_id = int(custom_data)
            elif "curveNumber" in first_point:
                curve_number = int(first_point["curveNumber"])
                if curve_number < len(df_parallel):
                    selected_line_id = curve_number

    if selected_line_id is not None and len(df_parallel) > 0 and selected_line_id in set(df_parallel["Line_ID"]):
        selected_row = df_parallel[df_parallel["Line_ID"] == selected_line_id].iloc[0]
        metric_view = mo.vstack(
            [
                mo.md("### Selected Line Values"),
                mo.md(
                    f"""
                    | Co | Cr | Fe | Ni |
                    |---:|---:|---:|---:|
                    | {selected_row['Co']:.6f} | {selected_row['Cr']:.6f} | {selected_row['Fe']:.6f} | {selected_row['Ni']:.6f} |

                    | T | G_LIQ | G_FCC | DeltaG |
                    |---:|---:|---:|---:|
                    | {selected_row['T']:.0f} K | {selected_row['G_LIQ'] / 1000:.6f} kJ/mol | {selected_row['G_FCC'] / 1000:.6f} kJ/mol | {selected_row['DeltaG'] / 1000:.6f} kJ/mol |
                    """
                ),
                pd.DataFrame([selected_row[variables]]),
            ],
            gap=1,
        )
    else:
        metric_view = mo.callout(
            "Hover over a colored line to see values in the Plotly tooltip. Use box/lasso selection on the plot or choose a Line_ID from the dropdown to print values here.",
            kind="info",
        )

    return metric_view,


@app.cell(hide_code=True)
def _(
    data_status,
    df_norm,
    df_parallel,
    interactive_plot,
    manual_line_id,
    metric_view,
    missing_status,
    mo,
    png_download,
    processing_status,
    show_normalized_data,
    show_raw_data,
    variables,
):
    output_items = [data_status]
    if missing_status is not None:
        output_items.append(missing_status)
    output_items.extend(
        [
            processing_status,
            mo.md("## Interactive Parallel Plot"),
            png_download,
            # interactive_plot,
            mo.md("## Hovered / Selected Line Values"),
            manual_line_id,
            metric_view,
        ]
    )

    if show_raw_data.value and len(df_parallel) > 0:
        output_items.extend([mo.md("## Raw Loaded Data"), df_parallel[variables]])

    if show_normalized_data.value and len(df_norm) > 0:
        output_items.extend([mo.md("## Normalized Plot Data"), df_norm])

    mo.vstack(output_items, gap=1)
    return


if __name__ == "__main__":
    app.run()
