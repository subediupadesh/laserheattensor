# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import re
    import io
    import json
    from typing import Dict, List, Optional, Tuple

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    MLDATA_URL = "https://subediupadesh.github.io/MultiComponentLaserAM/MLDATA"

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOCAL_MLDATA_DIR = os.path.join(SCRIPT_DIR, "MLDATA")

    if os.path.isdir(LOCAL_MLDATA_DIR):
        MLDATA_DIR = LOCAL_MLDATA_DIR
        DEFAULT_DIRS = {
            "TEMP": os.path.join(LOCAL_MLDATA_DIR, "TEMP"),
            "COMP": os.path.join(LOCAL_MLDATA_DIR, "COMP"),
            "ETALIQ": os.path.join(LOCAL_MLDATA_DIR, "ETALIQ"),
            "VEL": os.path.join(LOCAL_MLDATA_DIR, "VEL"),
        }
    else:
        MLDATA_DIR = MLDATA_URL
        DEFAULT_DIRS = {
            "TEMP": f"{MLDATA_URL}/TEMP",
            "COMP": f"{MLDATA_URL}/COMP",
            "ETALIQ": f"{MLDATA_URL}/ETALIQ",
            "VEL": f"{MLDATA_URL}/VEL",
        }

    ELEMENTS = ["Co", "Cr", "Fe", "Ni"]
    FALLBACK_CTF_TABLE: Dict[int, Dict[str, List[float]]] = {
        0: {"liquid": [0.35, 0.13, 0.15, 0.37], "fcc": [0.28, 0.18, 0.22, 0.32]},
        1: {"liquid": [0.32, 0.15, 0.17, 0.36], "fcc": [0.25, 0.21, 0.25, 0.29]},
        2: {"liquid": [0.30, 0.19, 0.11, 0.40], "fcc": [0.25, 0.23, 0.20, 0.32]},
        3: {"liquid": [0.33, 0.11, 0.23, 0.33], "fcc": [0.31, 0.15, 0.27, 0.27]},
        4: {"liquid": [0.38, 0.17, 0.19, 0.26], "fcc": [0.30, 0.22, 0.26, 0.22]},
    }

    CMAP_OPTIONS = [
        "inferno", "magma", "plasma", "viridis", "cividis", "turbo", "rainbow", "jet",
        "hot", "afmhot", "coolwarm", "RdBu_r", "RdYlBu_r", "Spectral_r", "PuBuGn",
        "YlOrRd", "YlGnBu", "GnBu", "Blues", "Greens", "gray", "binary", "bone",
        "terrain", "nipy_spectral",
    ]
    return (
        CMAP_OPTIONS,
        DEFAULT_DIRS,
        Dict,
        ELEMENTS,
        FALLBACK_CTF_TABLE,
        List,
        MLDATA_DIR,
        Optional,
        SCRIPT_DIR,
        Tuple,
        go,
        io,
        json,
        make_axes_locatable,
        mpl,
        mo,
        np,
        os,
        pd,
        plt,
        re,
    )


@app.cell
def _(ELEMENTS, FALLBACK_CTF_TABLE, SCRIPT_DIR, io, json, np, os, pd, re):
    def load_ctf_table_from_csv():
        csv_path = os.path.join(SCRIPT_DIR, "nomenclature_phase_composition_thermodynamic_factor_tensor_components.csv")
        if not os.path.exists(csv_path):
            return FALLBACK_CTF_TABLE
        try:
            df = pd.read_csv(csv_path)
            required_cols = {"cTF", "ETA", "Co", "Cr", "Fe", "Ni"}
            if not required_cols.issubset(df.columns):
                return FALLBACK_CTF_TABLE
            table = {}
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

    CTF_TABLE = load_ctf_table_from_csv()

    def parse_tensor_filename(filename: str):
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

    def standardize_field_array(arr, field_key: str):
        arr = np.asarray(arr)
        if field_key == "VEL":
            if arr.ndim == 4 and arr.shape[-1] in (2, 3):
                arr = np.linalg.norm(arr, axis=-1)
            elif arr.ndim == 4 and arr.shape[1] in (2, 3):
                arr = np.linalg.norm(arr, axis=1)
        return arr

    def get_component_view(arr, field_key: str, component_index: int):
        arr = standardize_field_array(arr, field_key)
        if arr.ndim == 3:
            return arr.astype(np.float64)
        if arr.ndim == 4:
            if arr.shape[-1] in (3, 4, 5):
                component_index = min(component_index, arr.shape[-1] - 1)
                return arr[..., component_index].astype(np.float64)
            if arr.shape[1] in (3, 4, 5):
                component_index = min(component_index, arr.shape[1] - 1)
                return arr[:, component_index, :, :].astype(np.float64)
        raise ValueError(f"Expected 3D scalar field or supported 4D component/vector field, got shape {arr.shape}.")

    async def load_sources_from_folder(folder_path: str, field_key: str):
        sources, arrays, warnings = [], [], []
        folder_path = str(folder_path).rstrip("/")
        is_web_path = folder_path.startswith("http://") or folder_path.startswith("https://")

        if is_web_path:
            try:
                from pyodide.http import open_url, pyfetch
            except ImportError as exc:
                return sources, arrays, [f"Web loading is only available in the browser/WASM export: {exc}"], 0

            base_mldata_url = folder_path.rsplit("/", 1)[0]
            manifest_url = f"{base_mldata_url}/manifest.json"

            try:
                manifest = json.loads(open_url(manifest_url).read())
                files = sorted(manifest.get(field_key, []))
            except Exception as exc:
                return sources, arrays, [f"Could not read manifest: {manifest_url}; {exc}"], 0

            for filename in files:
                parsed = parse_tensor_filename(filename)
                if parsed is None:
                    warnings.append(f"Skipped {filename}: filename must be p<P>s<v>cTF<idx>.npy")
                    continue

                file_url = f"{folder_path}/{filename}"

                try:
                    response = await pyfetch(file_url)

                    if not response.ok:
                        warnings.append(f"Skipped {filename}: HTTP {response.status} from {file_url}")
                        continue

                    data = await response.bytes()
                    arr = np.load(io.BytesIO(data), allow_pickle=False)
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

        if not os.path.isdir(folder_path):
            return sources, arrays, [f"Folder not found: {folder_path}"], 0

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".npy")])

        for filename in files:
            parsed = parse_tensor_filename(filename)
            if parsed is None:
                warnings.append(f"Skipped {filename}: filename must be p<P>s<v>cTF<idx>.npy")
                continue

            try:
                arr = np.load(os.path.join(folder_path, filename), allow_pickle=False)
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

    def sources_dataframe(sources):
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

    def mpl_to_plotly_colorscale(cmap_name: str, n: int = 255):
        cmap = mpl.colormaps.get_cmap(cmap_name)
        return [[i / (n - 1), mpl.colors.rgb2hex(cmap(i / (n - 1)))] for i in range(n)]

    return (
        CTF_TABLE,
        get_component_view,
        load_ctf_table_from_csv,
        load_sources_from_folder,
        mpl_to_plotly_colorscale,
        parse_tensor_filename,
        sources_dataframe,
        standardize_field_array,
    )


@app.cell
def _(np):
    class PhaseAwareAttentionInterpolator:
        def __init__(self, sigma_param=0.20, sigma_comp=0.05, num_heads=4, d_head=8, comp_strength=1.0, seed=42):
            self.sigma_param = float(sigma_param)
            self.sigma_comp = float(sigma_comp)
            self.num_heads = int(num_heads)
            self.d_head = int(d_head)
            self.comp_strength = float(comp_strength)
            rng = np.random.default_rng(int(seed))
            scale = 1.0 / np.sqrt(2.0)
            self.W_q_proc = rng.normal(0.0, scale, size=(self.num_heads * self.d_head, 2))
            self.W_k_proc = rng.normal(0.0, scale, size=(self.num_heads * self.d_head, 2))

        @staticmethod
        def normalize_params(params_list, target_params):
            all_params = np.array(list(params_list) + [target_params], dtype=float)
            mins = all_params.min(axis=0)
            maxs = all_params.max(axis=0)
            ranges = maxs - mins + 1e-8
            src = (np.array(params_list, dtype=float) - mins) / ranges
            tgt = (np.array(target_params, dtype=float) - mins) / ranges
            return src, tgt

        @staticmethod
        def phase_composition_vector(item):
            liquid = np.asarray(item.get("composition_liquid", item.get("composition")), dtype=float)
            fcc = np.asarray(item.get("composition_fcc", liquid), dtype=float)
            return np.concatenate([liquid, fcc])

        def composition_similarity(self, source_comp, target_comp):
            diff2 = float(np.sum((np.asarray(source_comp) - np.asarray(target_comp)) ** 2))
            return float(np.exp(-diff2 / (2.0 * self.sigma_comp ** 2)))

        @staticmethod
        def softmax(x, axis=0):
            x = np.asarray(x, dtype=float)
            x = x - np.max(x, axis=axis, keepdims=True)
            ex = np.exp(x)
            return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)

        def compute_weights(self, sources, target, use_composition=True):
            n = len(sources)
            source_proc = [(s["P"], s["v"]) for s in sources]
            target_proc = (target["P"], target["v"])
            src_proc, tgt_proc = self.normalize_params(source_proc, target_proc)

            q = (self.W_q_proc @ tgt_proc).reshape(self.num_heads, self.d_head)
            k = (src_proc @ self.W_k_proc.T).reshape(n, self.num_heads, self.d_head)
            logits = np.einsum("nhd,hd->nh", k, q) / np.sqrt(self.d_head)
            attn_by_head = self.softmax(logits, axis=0)
            attn = attn_by_head.mean(axis=1)
            attn = attn / (attn.sum() + 1e-12)

            dists = np.sqrt(np.sum((src_proc - tgt_proc[None, :]) ** 2, axis=1))
            gaussian = np.exp(-(dists ** 2) / (2.0 * self.sigma_param ** 2))
            gaussian = gaussian / (gaussian.sum() + 1e-12)

            if use_composition:
                target_comp_vec = self.phase_composition_vector(target)
                sims = np.array([
                    self.composition_similarity(self.phase_composition_vector(s), target_comp_vec)
                    for s in sources
                ], dtype=float)
                sims = np.power(sims + 1e-12, self.comp_strength)
                comp = sims / (sims.sum() + 1e-12)
            else:
                comp = np.ones(n, dtype=float) / n

            combined = attn * gaussian * comp
            combined = combined / (combined.sum() + 1e-12)
            return {
                "attention_weights_proc": attn,
                "spatial_weights_proc": gaussian,
                "composition_weights": comp,
                "combined_weights": combined,
                "norm_sources_proc": src_proc,
                "norm_target_proc": tgt_proc,
                "W_q_proc": self.W_q_proc,
                "W_k_proc": self.W_k_proc,
            }
    return PhaseAwareAttentionInterpolator,


@app.cell
def _(CTF_TABLE, DEFAULT_DIRS, ELEMENTS, mo, np, os, pd, CMAP_OPTIONS):
    title = mo.md(
        """
        # Phase-Aware Attention Interpolation for TEMP, ETALIQ, VEL and COMP

        This marimo version reads `.npy` simulation files from local folders such as
        `MLDATA/TEMP`, `MLDATA/ETALIQ`, `MLDATA/VEL`, and `MLDATA/COMP`.
        Source filenames must follow `p<P>s<v>cTF<k>.npy`, for example `p350s45cTF0.npy`.
        """
    )

    field_select = mo.ui.dropdown(
        options=["TEMP", "ETALIQ", "VEL", "COMP", "COMBO"],
        value="TEMP",
        label="Prediction tab",
    )
    p_target = mo.ui.number(start=300, stop=800, step=10, value=370, label="Power, P (W)")
    v_target_cm_s = mo.ui.number(start=40, stop=80, step=5, value=50, label="Scan speed, v (cm/s)")
    ctf_options = [f"cTF{ctf_key}" for ctf_key in sorted(CTF_TABLE.keys())] + ["cTF_New"]
    target_ctf_choice = mo.ui.dropdown(options=ctf_options, value="cTF0", label="Target composition set")

    liq_co = mo.ui.slider(0.0, 1.0, step=0.01, value=0.35, label="Custom LIQ Co")
    liq_cr = mo.ui.slider(0.0, 1.0, step=0.01, value=0.13, label="Custom LIQ Cr")
    liq_fe = mo.ui.slider(0.0, 1.0, step=0.01, value=0.15, label="Custom LIQ Fe")
    fcc_co = mo.ui.slider(0.0, 1.0, step=0.01, value=0.28, label="Custom FCC Co")
    fcc_cr = mo.ui.slider(0.0, 1.0, step=0.01, value=0.18, label="Custom FCC Cr")
    fcc_fe = mo.ui.slider(0.0, 1.0, step=0.01, value=0.22, label="Custom FCC Fe")

    sigma_param = mo.ui.slider(0.05, 0.50, step=0.01, value=0.18, label="Gaussian locality σ for P-v")
    sigma_comp = mo.ui.slider(0.01, 0.25, step=0.01, value=0.05, label="Composition similarity σ")
    comp_strength = mo.ui.slider(0.0, 2.0, step=0.1, value=1.0, label="Composition influence strength")
    num_heads = mo.ui.slider(1, 12, step=1, value=8, label="Attention heads")
    d_head = mo.ui.slider(4, 16, step=1, value=6, label="Dimension per head")
    seed = mo.ui.number(start=0, stop=9999, step=1, value=42, label="Random seed")
    use_composition = mo.ui.checkbox(value=True, label="Use composition similarity in weighting")

    cmap_default = {"TEMP": "rainbow", "ETALIQ": "RdBu_r", "VEL": "turbo", "COMP": "viridis", "COMBO": "turbo"}
    cmap_choice = mo.ui.dropdown(options=CMAP_OPTIONS, value="rainbow", label="Main colormap")
    temp_cmap = mo.ui.dropdown(options=CMAP_OPTIONS, value="rainbow", label="TEMP colormap")
    phase_cmap = mo.ui.dropdown(options=CMAP_OPTIONS, value="RdBu_r", label="ETALIQ colormap")
    vel_cmap = mo.ui.dropdown(options=CMAP_OPTIONS, value="turbo", label="VEL colormap")
    comp_cmap = mo.ui.dropdown(options=CMAP_OPTIONS, value="viridis", label="COMP colormap")
    phase_threshold = mo.ui.slider(0.01, 0.99, step=0.01, value=0.50, label="LIQUID threshold")

    folder_temp = mo.ui.text(value=DEFAULT_DIRS["TEMP"], label="TEMP folder")
    folder_eta = mo.ui.text(value=DEFAULT_DIRS["ETALIQ"], label="ETALIQ folder")
    folder_vel = mo.ui.text(value=DEFAULT_DIRS["VEL"], label="VEL folder")
    folder_comp = mo.ui.text(value=DEFAULT_DIRS["COMP"], label="COMP folder")
    component_choice = mo.ui.dropdown(options=ELEMENTS, value="Co", label="COMP/component display")
    time_index = mo.ui.number(start=0, stop=999999, step=1, value=0, label="Static time index")
    run_button = mo.ui.run_button(label="Run prediction")

    controls = mo.vstack([
        title,
        mo.md("## Controls"),
        mo.hstack([field_select, p_target, v_target_cm_s, target_ctf_choice], justify="start"),
        mo.accordion({
            "Custom composition, used only when cTF_New is selected": mo.vstack([
                mo.md("**LIQUID**"), mo.hstack([liq_co, liq_cr, liq_fe]),
                mo.md("**FCC**"), mo.hstack([fcc_co, fcc_cr, fcc_fe]),
            ]),
            "Attention model": mo.hstack([sigma_param, sigma_comp, comp_strength, num_heads, d_head, seed, use_composition]),
            "Folders": mo.vstack([folder_temp, folder_eta, folder_vel, folder_comp]),
            "Plot settings": mo.hstack([cmap_choice, temp_cmap, phase_cmap, vel_cmap, comp_cmap, phase_threshold, component_choice, time_index]),
        }),
        run_button,
    ])
    controls
    return (
        cmap_choice,
        comp_cmap,
        comp_strength,
        component_choice,
        controls,
        ctf_options,
        d_head,
        fcc_co,
        fcc_cr,
        fcc_fe,
        field_select,
        folder_comp,
        folder_eta,
        folder_temp,
        folder_vel,
        liq_co,
        liq_cr,
        liq_fe,
        num_heads,
        p_target,
        phase_cmap,
        phase_threshold,
        run_button,
        seed,
        sigma_comp,
        sigma_param,
        target_ctf_choice,
        temp_cmap,
        time_index,
        use_composition,
        v_target_cm_s,
        vel_cmap,
    )


@app.cell
def _(
    CTF_TABLE,
    ELEMENTS,
    fcc_co,
    fcc_cr,
    fcc_fe,
    liq_co,
    liq_cr,
    liq_fe,
    mo,
    np,
    p_target,
    pd,
    target_ctf_choice,
    v_target_cm_s,
):
    if target_ctf_choice.value != "cTF_New":
        selected_ctf = int(str(target_ctf_choice.value).replace("cTF", ""))
        target_composition = np.array(CTF_TABLE[selected_ctf]["liquid"], dtype=float)
        target_composition_fcc = np.array(CTF_TABLE[selected_ctf]["fcc"], dtype=float)
        target_valid = True
        comp_info = mo.vstack([
            mo.md("### Selected phase-specific composition"),
            mo.ui.table(pd.DataFrame([dict(zip(ELEMENTS, target_composition))], index=[f"{target_ctf_choice.value} LIQ"])),
            mo.ui.table(pd.DataFrame([dict(zip(ELEMENTS, target_composition_fcc))], index=[f"{target_ctf_choice.value} FCC"])),
        ])
    else:
        ni_target = round(1.0 - float(liq_co.value) - float(liq_cr.value) - float(liq_fe.value), 4)
        ni_fcc_target = round(1.0 - float(fcc_co.value) - float(fcc_cr.value) - float(fcc_fe.value), 4)
        target_composition = np.array([float(liq_co.value), float(liq_cr.value), float(liq_fe.value), ni_target], dtype=float)
        target_composition_fcc = np.array([float(fcc_co.value), float(fcc_cr.value), float(fcc_fe.value), ni_fcc_target], dtype=float)
        target_valid = (0.0 <= ni_target <= 1.0) and (0.0 <= ni_fcc_target <= 1.0)
        status = "✅ valid" if target_valid else "❌ invalid: adjust Co, Cr or Fe"
        comp_info = mo.vstack([
            mo.md(f"### Custom composition: {status}"),
            mo.md(f"LIQ Ni = `{ni_target:.4f}`, FCC Ni = `{ni_fcc_target:.4f}`"),
            mo.ui.table(pd.DataFrame([dict(zip(ELEMENTS, target_composition))], index=["LIQ"])),
            mo.ui.table(pd.DataFrame([dict(zip(ELEMENTS, target_composition_fcc))], index=["FCC"])),
        ])

    TARGET = {
        "P": float(p_target.value),
        "v_cm_s": float(v_target_cm_s.value),
        "v": float(v_target_cm_s.value) / 100.0,
        "composition": target_composition,
        "composition_liquid": target_composition,
        "composition_fcc": target_composition_fcc,
        "ctf_choice": str(target_ctf_choice.value),
    }
    comp_info
    return TARGET, comp_info, target_composition, target_composition_fcc, target_valid


@app.cell
def _(
    PhaseAwareAttentionInterpolator,
    TARGET,
    comp_strength,
    d_head,
    get_component_view,
    load_sources_from_folder,
    np,
    num_heads,
    phase_threshold,
    seed,
    sigma_comp,
    sigma_param,
    use_composition,
):
    async def compute_prediction(field_key: str, folder_path: str, component_index: int = 0):
        sources, arrays_raw, warnings, file_count = await load_sources_from_folder(folder_path, field_key)
        if len(sources) < 2:
            return {
                "ok": False,
                "field_key": field_key,
                "file_count": file_count,
                "sources": sources,
                "warnings": warnings,
                "message": f"{field_key}: need at least two valid source files.",
            }

        arrays, shapes = [], []
        for src, arr in zip(sources, arrays_raw):
            try:
                scalar = get_component_view(arr, field_key, component_index)
                arrays.append(scalar)
                shapes.append(scalar.shape)
            except Exception as exc:
                warnings.append(f"{src['file_name']}: {exc}")

        if len(arrays) < 2:
            return {"ok": False, "field_key": field_key, "file_count": file_count, "sources": sources, "warnings": warnings, "message": f"{field_key}: not enough valid scalar fields."}
        if len(set(shapes)) != 1:
            return {"ok": False, "field_key": field_key, "file_count": file_count, "sources": sources, "warnings": warnings, "message": f"{field_key}: scalar field shapes do not match: {sorted(set(shapes))}"}

        model = PhaseAwareAttentionInterpolator(
            sigma_param=float(sigma_param.value),
            sigma_comp=float(sigma_comp.value),
            num_heads=int(num_heads.value),
            d_head=int(d_head.value),
            comp_strength=float(comp_strength.value),
            seed=int(seed.value),
        )
        results = model.compute_weights(sources, TARGET, use_composition=bool(use_composition.value))
        weights = results["combined_weights"]
        predicted = np.zeros_like(arrays[0], dtype=np.float64)
        for weight, arr in zip(weights, arrays):
            predicted += float(weight) * arr
        if field_key == "ETALIQ":
            predicted = (predicted >= float(phase_threshold.value)).astype(np.float64)
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
    return compute_prediction,


@app.cell
def _(component_choice, field_select, folder_comp, folder_eta, folder_temp, folder_vel):
    field_key = str(field_select.value)
    folder_by_field = {
        "TEMP": folder_temp.value,
        "ETALIQ": folder_eta.value,
        "VEL": folder_vel.value,
        "COMP": folder_comp.value,
    }
    component_index_map = {"Co": 0, "Cr": 1, "Fe": 2, "Ni": 3}
    component_index = component_index_map.get(str(component_choice.value), 0)
    return component_index, component_index_map, field_key, folder_by_field


@app.cell
async def _(
    compute_prediction,
    component_index,
    field_key,
    folder_by_field,
    folder_comp,
    folder_eta,
    folder_temp,
    folder_vel,
    run_button,
    target_valid,
):
    prediction_result = None
    combo_result = None
    if run_button.value and target_valid:
        if field_key == "COMBO":
            combo_result = {
                "TEMP": await compute_prediction("TEMP", folder_temp.value, 0),
                "ETALIQ": await compute_prediction("ETALIQ", folder_eta.value, 0),
                "VEL": await compute_prediction("VEL", folder_vel.value, 0),
            }
        else:
            prediction_result = await compute_prediction(field_key, folder_by_field[field_key], component_index)
    return combo_result, prediction_result


@app.cell
def _(combo_result, field_key, mo, prediction_result, target_valid):
    if not target_valid:
        status_message = mo.md("❌ Target composition is invalid. Select another cTF or adjust custom composition.")
    elif not prediction_result and not combo_result:
        status_message = mo.md("Click **Run prediction** after selecting folders and parameters.")
    elif field_key == "COMBO" and combo_result:
        status_message = mo.md("COMBO prediction requested.")
    elif prediction_result and not prediction_result.get("ok", False):
        status_message = mo.md(f"❌ {prediction_result.get('message', 'Prediction failed.')}")
    else:
        status_message = mo.md("✅ Prediction calculated.")
    status_message
    return status_message,


@app.cell
def _(mo, prediction_result, sources_dataframe):
    if prediction_result and prediction_result.get("sources"):
        warnings = prediction_result.get("warnings", [])
        warning_text = "\n".join([f"- {w}" for w in warnings[:10]]) if warnings else "No loading warnings."
        source_summary = mo.accordion({
            "Loaded source simulations": mo.ui.table(sources_dataframe(prediction_result["sources"])),
            "Loading warnings": mo.md(warning_text),
        })
    else:
        source_summary = mo.md("")
    source_summary
    return source_summary,


@app.cell
def _(go, np, pd, prediction_result):
    weights_table = None
    param_fig = None
    qk_fig = None
    if prediction_result and prediction_result.get("ok", False):
        sources = prediction_result["sources"]
        res = prediction_result["results"]
        weights_table = pd.DataFrame({
            "Source File": [s["file_name"] for s in sources],
            "P (W)": [s["P"] for s in sources],
            "v (cm/s)": [s["v_cm_s"] for s in sources],
            "cTF": [s["TF_idx"] for s in sources],
            "Process attention": np.round(res["attention_weights_proc"], 6),
            "P-v Gaussian": np.round(res["spatial_weights_proc"], 6),
            "Composition similarity": np.round(res["composition_weights"], 6),
            "Final weight": np.round(res["combined_weights"], 6),
        })

        src_norm = res["norm_sources_proc"]
        tgt_norm = res["norm_target_proc"]
        param_fig = go.Figure()
        param_fig.add_trace(go.Scatter(
            x=src_norm[:, 0],
            y=src_norm[:, 1],
            mode="markers+text",
            marker=dict(size=16, color=res["combined_weights"], colorscale="Viridis", showscale=True, colorbar=dict(title="Final weight")),
            text=[f"p{s['P']} s{s['v_cm_s']} cTF{s['TF_idx']}" for s in sources],
            textposition="top center",
            name="Sources",
        ))
        param_fig.add_trace(go.Scatter(
            x=[tgt_norm[0]],
            y=[tgt_norm[1]],
            mode="markers+text",
            marker=dict(size=24, color="red", symbol="star"),
            text=["Target"],
            textposition="bottom center",
            name="Target",
        ))
        param_fig.update_layout(
            title="Normalized parameter space",
            xaxis_title="Normalized power",
            yaxis_title="Normalized scan speed",
            height=520,
        )
    return param_fig, qk_fig, weights_table


@app.cell
def _(mo, param_fig, prediction_result, weights_table):
    if prediction_result and prediction_result.get("ok", False):
        weights_view = mo.vstack([
            mo.md("## Hybrid attention weights"),
            mo.ui.table(weights_table),
            param_fig,
        ])
    else:
        weights_view = mo.md("")
    weights_view
    return weights_view,


@app.cell
def _(go, mpl_to_plotly_colorscale, np):
    def _thin_for_marimo_plot(field_pred, max_side=220, max_frames=10):
        _arr = np.asarray(field_pred, dtype=float)
        _nt, _ny, _nx = _arr.shape
        _t_stride = max(1, int(np.ceil(_nt / max_frames)))
        _xy_stride = max(1, int(np.ceil(max(_ny, _nx) / max_side)))
        _thin = _arr[::_t_stride, ::_xy_stride, ::_xy_stride]
        _time_labels = list(range(0, _nt, _t_stride))
        return _thin, _time_labels, _t_stride, _xy_stride

    def plot_field_animation(field_pred, cmap_name, title, colorbar_title, is_phase=False):
        original_shape = np.asarray(field_pred).shape
        field_pred, time_labels, t_stride, xy_stride = _thin_for_marimo_plot(field_pred)
        Nt = field_pred.shape[0]
        colorscale = mpl_to_plotly_colorscale(cmap_name)
        global_min = 0.0 if is_phase else float(np.nanmin(field_pred))
        global_max = 1.0 if is_phase else float(np.nanmax(field_pred))
        if np.isclose(global_min, global_max):
            global_max = global_min + 1e-12
        ticks = [0.0, 1.0] if is_phase else np.linspace(global_min, global_max, 5).tolist()
        ticktext = ["FCC = 0", "LIQUID = 1"] if is_phase else [f"{v:.4g}" for v in ticks]

        def heat(k):
            return go.Heatmap(
                z=np.flipud(field_pred[k]),
                colorscale=colorscale,
                zmin=global_min,
                zmax=global_max,
                colorbar=dict(title=colorbar_title, tickmode="array", tickvals=ticks, ticktext=ticktext),
                hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4g}<extra></extra>",
            )

        frames = [go.Frame(data=[heat(k)], name=f"t{k}") for k in range(Nt)]
        fig = go.Figure(data=[heat(0)], frames=frames)
        _display_note = f"displayed as {field_pred.shape} from full {original_shape}; time stride={t_stride}, spatial stride={xy_stride}"
        fig.update_layout(
            title=f"{title}<br><sup>{_display_note}</sup>",
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title="", scaleanchor="x", scaleratio=1),
            updatemenus=[{
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
                "type": "buttons",
                "x": 0.05,
                "y": -0.05,
            }],
            sliders=[{
                "active": 0,
                "steps": [
                    {"method": "animate", "args": [[f"t{k}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], "label": str(time_labels[k])}
                    for k in range(Nt)
                ],
                "x": 0.1,
                "len": 0.85,
                "y": -0.05,
                "currentvalue": {"prefix": "Time index: "},
            }],
            margin=dict(l=0, r=0, t=55, b=80),
            height=720,
        )
        return fig
    return plot_field_animation,


@app.cell
def _(cmap_choice, comp_cmap, field_key, np, phase_cmap, plot_field_animation, prediction_result, temp_cmap, vel_cmap):
    field_fig = None
    stats_text = ""
    if prediction_result and prediction_result.get("ok", False):
        _pred = prediction_result["prediction"]
        cmap_by_field = {
            "TEMP": temp_cmap.value,
            "ETALIQ": phase_cmap.value,
            "VEL": vel_cmap.value,
            "COMP": comp_cmap.value,
        }
        unit_by_field = {
            "TEMP": "K",
            "ETALIQ": "phase",
            "VEL": "velocity magnitude",
            "COMP": "mole fraction",
        }
        is_phase = field_key == "ETALIQ"
        field_fig = plot_field_animation(
            _pred,
            cmap_by_field.get(field_key, cmap_choice.value),
            f"{field_key} prediction",
            unit_by_field.get(field_key, "value"),
            is_phase=is_phase,
        )
        stats_text = (
            f"Shape = {_pred.shape}; min = {float(np.nanmin(_pred)):.6g}; "
            f"max = {float(np.nanmax(_pred)):.6g}; mean = {float(np.nanmean(_pred)):.6g}"
        )
    return field_fig, stats_text


@app.cell
def _(field_fig, mo, stats_text):
    if field_fig is not None:
        field_view = mo.vstack([mo.md("## Predicted field animation"), mo.md(stats_text), field_fig])
    else:
        field_view = mo.md("")
    field_view
    return field_view,


@app.cell
def _(io, np, prediction_result):
    full_npz_bytes = None
    download_note = ""
    if prediction_result and prediction_result.get("ok", False):
        download_pred = prediction_result["prediction"]
        buf_npz = io.BytesIO()
        np.savez_compressed(
            buf_npz,
            field=download_pred,
            weights=prediction_result["results"]["combined_weights"],
        )
        _candidate_npz = buf_npz.getvalue()
        if len(_candidate_npz) <= 8_000_000:
            full_npz_bytes = _candidate_npz
        else:
            download_note = (
                f"Compressed prediction is {len(_candidate_npz) / 1_000_000:.1f} MB, "
                "so the in-browser download button is hidden to avoid marimo output-size errors. "
                "Use the app output visually, or save the prediction from a script if you need the full array."
            )
    return download_note, full_npz_bytes


@app.cell
def _(download_note, field_key, full_npz_bytes, mo, p_target, v_target_cm_s):
    if full_npz_bytes is not None:
        base = f"{field_key}_pred_p{int(float(p_target.value))}s{int(float(v_target_cm_s.value))}"
        downloads = mo.download(data=full_npz_bytes, filename=f"{base}.npz", mimetype="application/octet-stream", label="Download compressed .npz with weights")
    elif download_note:
        downloads = mo.md(download_note)
    else:
        downloads = mo.md("")
    downloads
    return downloads,


@app.cell
def _(combo_result, go, mpl_to_plotly_colorscale, np, phase_cmap, temp_cmap, vel_cmap):
    def _thin_combo_for_marimo_plot(_temp, _eta, _vel, max_side=150, max_frames=8):
        _nt, _ny, _nx = _temp.shape
        _t_stride = max(1, int(np.ceil(_nt / max_frames)))
        _xy_stride = max(1, int(np.ceil(max(_ny, _nx) / max_side)))
        _time_labels = list(range(0, _nt, _t_stride))
        return (
            _temp[::_t_stride, ::_xy_stride, ::_xy_stride],
            _eta[::_t_stride, ::_xy_stride, ::_xy_stride],
            _vel[::_t_stride, ::_xy_stride, ::_xy_stride],
            _time_labels,
            _t_stride,
            _xy_stride,
        )

    def make_combo_fig(combo):
        temp_pred = np.asarray(combo["TEMP"]["prediction"], dtype=float)
        eta_pred = (np.asarray(combo["ETALIQ"]["prediction"], dtype=float) >= 0.5).astype(float)
        vel_pred = np.asarray(combo["VEL"]["prediction"], dtype=float)
        if temp_pred.shape != eta_pred.shape or temp_pred.shape != vel_pred.shape:
            raise ValueError(f"Shapes must match: TEMP={temp_pred.shape}, ETALIQ={eta_pred.shape}, VEL={vel_pred.shape}")
        original_shape = temp_pred.shape
        temp_pred, eta_pred, vel_pred, time_labels, t_stride, xy_stride = _thin_combo_for_marimo_plot(temp_pred, eta_pred, vel_pred)
        Nt = temp_pred.shape[0]
        liquid_mask = eta_pred >= 0.5
        vel_masked = np.where(liquid_mask, vel_pred, 0.0)
        from plotly.subplots import make_subplots

        def frame_min_max_ticks(arr, n_ticks=5):
            finite = np.asarray(arr)[np.isfinite(arr)]
            if finite.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-12
            return vmin, vmax, np.linspace(vmin, vmax, n_ticks)

        def traces(k):
            temp = np.flipud(temp_pred[k])
            eta = np.flipud(eta_pred[k])
            vel = np.flipud(vel_masked[k])
            tmin, tmax, tticks = frame_min_max_ticks(temp)
            vmin, vmax, vticks = frame_min_max_ticks(vel)
            return [
                go.Heatmap(z=temp, colorscale=mpl_to_plotly_colorscale(temp_cmap.value), zmin=tmin, zmax=tmax,
                           colorbar=dict(title="Temperature (K)", x=1.02, y=0.85, len=0.24, tickvals=tticks, ticktext=[f"{x:.4g}" for x in tticks]), showscale=True),
                go.Heatmap(z=eta, colorscale=mpl_to_plotly_colorscale(phase_cmap.value), zmin=0, zmax=1,
                           colorbar=dict(title="ETALIQ", x=1.02, y=0.51, len=0.24, tickvals=[0, 1], ticktext=["FCC", "LIQ"]), showscale=True),
                go.Contour(z=eta, contours=dict(coloring="lines", start=0.5, end=0.5, size=1), line=dict(color="white", width=3), showscale=False, hoverinfo="skip"),
                go.Heatmap(z=vel, colorscale=mpl_to_plotly_colorscale(vel_cmap.value), zmin=vmin, zmax=vmax,
                           colorbar=dict(title="Velocity", x=1.02, y=0.17, len=0.24, tickvals=vticks, ticktext=[f"{x:.4g}" for x in vticks]), showscale=True),
                go.Contour(z=eta, contours=dict(coloring="lines", start=0.5, end=0.5, size=1), line=dict(color="white", width=3), showscale=False, hoverinfo="skip"),
            ]

        fig = make_subplots(rows=3, cols=1, vertical_spacing=0.065, subplot_titles=("Temperature", "ETALIQ", "Velocity masked by LIQUID"))
        init = traces(0)
        fig.add_trace(init[0], row=1, col=1)
        fig.add_trace(init[1], row=2, col=1)
        fig.add_trace(init[2], row=2, col=1)
        fig.add_trace(init[3], row=3, col=1)
        fig.add_trace(init[4], row=3, col=1)
        fig.frames = [go.Frame(data=traces(frame_idx), traces=[0, 1, 2, 3, 4], name=f"t{frame_idx}") for frame_idx in range(Nt)]
        fig.update_layout(
            title=f"COMBO Prediction<br><sup>displayed as {temp_pred.shape} from full {original_shape}; time stride={t_stride}, spatial stride={xy_stride}</sup>",
            height=1700,
            margin=dict(l=25, r=120, t=90, b=120),
            updatemenus=[{"buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ], "type": "buttons", "x": 0, "y": -0.04}],
            sliders=[{"active": 0, "steps": [
                {"method": "animate", "args": [[f"t{frame_idx}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], "label": str(time_labels[frame_idx])}
                for frame_idx in range(Nt)
            ], "x": 0.12, "len": 0.86, "y": -0.05, "currentvalue": {"prefix": "Time index: "}}],
        )
        for axis_name in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
            fig.layout[axis_name].showticklabels = False
            fig.layout[axis_name].title = ""
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=2, col=1)
        fig.update_yaxes(scaleanchor="x3", scaleratio=1, row=3, col=1)
        return fig

    combo_fig = None
    combo_error = ""
    if combo_result:
        try:
            if all(combo_result.get(combo_key, {}).get("ok", False) for combo_key in ["TEMP", "ETALIQ", "VEL"]):
                combo_fig = make_combo_fig(combo_result)
            else:
                messages = [combo_result.get(combo_key, {}).get("message", f"{combo_key} failed") for combo_key in ["TEMP", "ETALIQ", "VEL"] if not combo_result.get(combo_key, {}).get("ok", False)]
                combo_error = "\n".join(messages)
        except Exception as exc:
            combo_error = str(exc)
    return combo_error, combo_fig, make_combo_fig


@app.cell
def _(combo_error, combo_fig, combo_result, mo):
    if combo_fig is not None:
        combo_view = mo.vstack([mo.md("## COMBO synchronized animation"), combo_fig])
    elif combo_result and combo_error:
        combo_view = mo.md(f"❌ COMBO failed:\n\n{combo_error}")
    else:
        combo_view = mo.md("")
    combo_view
    return combo_view,


@app.cell
def _(mo):
    mo.accordion({
        "Run instructions": mo.md(
            """
            **Run locally**

            ```bash
            marimo run Ablation_Calc_Marimo.py
            ```

            For local use, keep this file beside the `MLDATA/` folder. For GitHub Pages, the app automatically uses the deployed `MLDATA/manifest.json` and `.npy` URLs. The expected local structure is:

            ```text
            Ablation_Calc_Marimo.py
            MLDATA/
              TEMP/p350s45cTF0.npy
              ETALIQ/p350s45cTF0.npy
              VEL/p350s45cTF0.npy
              COMP/p350s45cTF0.npy
            ```

            Optional composition CSV beside the script:

            ```text
            nomenclature_phase_composition_thermodynamic_factor_tensor_components.csv
            ```
            """
        ),
        "Calculation used": mo.md(
            """
            1. Parse `P`, `v`, and `cTF` from `p<P>s<v>cTF<k>.npy`.
            2. Normalize `(P, v)` for the source and target simulations.
            3. Compute query/key attention weights with deterministic NumPy matrices.
            4. Multiply by Gaussian locality in `(P, v)` space.
            5. Multiply by phase-aware composition similarity using LIQUID and FCC `[Co, Cr, Fe, Ni]` vectors.
            6. Normalize the hybrid weights and interpolate the full `(Nt, Ny, Nx)` field.
            """
        ),
    })
    return


if __name__ == "__main__":
    app.run()
