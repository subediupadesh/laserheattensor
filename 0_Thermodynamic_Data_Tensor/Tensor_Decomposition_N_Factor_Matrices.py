import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import linalg

st.set_page_config(
    page_title="CPD Factor Matrices Only",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_DIR = os.path.join(SCRIPT_DIR, "csv_files")


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(csv_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(csv_dir, "Gibbs_*K.csv")))
    if not files:
        st.error(
            f"No Gibbs CSV files found in `{csv_dir}`.\n\n"
            "Expected files such as `Gibbs_300K.csv`, `Gibbs_700K.csv`, `Gibbs_800K.csv`, ..."
        )
        st.stop()

    frames = []
    for path in files:
        name = os.path.basename(path)
        try:
            T = int(name.replace("Gibbs_", "").replace("K.csv", ""))
            tmp = pd.read_csv(path, usecols=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"])
            tmp["T"] = T
            frames.append(tmp)
        except Exception as exc:
            st.warning(f"Skipping `{name}`: {exc}")

    if not frames:
        st.error("No valid Gibbs CSV files could be loaded.")
        st.stop()

    df = pd.concat(frames, ignore_index=True)
    for col in ["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC", "T"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC", "T"]).copy()
    df["T"] = df["T"].astype(int)
    return df


@st.cache_data(ttl=7200, show_spinner=False)
def build_tensor_data(df: pd.DataFrame):
    co_vals = np.array(sorted(df["Co"].unique()), dtype=float)
    cr_vals = np.array(sorted(df["Cr"].unique()), dtype=float)
    fe_vals = np.array(sorted(df["Fe"].unique()), dtype=float)
    T_vals = np.array(sorted(df["T"].unique()), dtype=float)

    n_co, n_cr, n_fe, n_T = len(co_vals), len(cr_vals), len(fe_vals), len(T_vals)

    co_to_idx = {round(float(v), 4): i for i, v in enumerate(co_vals)}
    cr_to_idx = {round(float(v), 4): i for i, v in enumerate(cr_vals)}
    fe_to_idx = {round(float(v), 4): i for i, v in enumerate(fe_vals)}
    T_to_idx = {int(v): i for i, v in enumerate(T_vals)}

    G_LIQ = np.full((n_co, n_cr, n_fe, n_T), np.nan, dtype=np.float64)
    G_FCC = np.full((n_co, n_cr, n_fe, n_T), np.nan, dtype=np.float64)

    for row in df.itertuples(index=False):
        i = co_to_idx.get(round(float(row.Co), 4))
        j = cr_to_idx.get(round(float(row.Cr), 4))
        k = fe_to_idx.get(round(float(row.Fe), 4))
        t = T_to_idx.get(int(row.T))
        if i is not None and j is not None and k is not None and t is not None:
            G_LIQ[i, j, k, t] = float(row.G_LIQ)
            G_FCC[i, j, k, t] = float(row.G_FCC)

    return {
        "G_LIQ": G_LIQ,
        "G_FCC": G_FCC,
        "dims": (n_co, n_cr, n_fe, n_T),
        "co_vals": co_vals,
        "cr_vals": cr_vals,
        "fe_vals": fe_vals,
        "T_vals": T_vals,
        "valid_count": int(np.isfinite(G_LIQ).sum()),
    }


def normalize_tensor(tensor: np.ndarray):
    finite = np.isfinite(tensor)
    if finite.sum() == 0:
        raise ValueError("Tensor contains no finite Gibbs-energy values.")
    mean = float(np.nanmean(tensor))
    std = float(np.nanstd(tensor))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return (tensor - mean) / std, mean, std


def unfold_tensor(tensor, mode):
    if mode == 0:
        return tensor.reshape(tensor.shape[0], -1)
    if mode == 1:
        return tensor.transpose(1, 0, 2, 3).reshape(tensor.shape[1], -1)
    if mode == 2:
        return tensor.transpose(2, 0, 1, 3).reshape(tensor.shape[2], -1)
    if mode == 3:
        return tensor.transpose(3, 0, 1, 2).reshape(tensor.shape[3], -1)
    raise ValueError("mode must be 0, 1, 2, or 3")


def _safe_columns(M: np.ndarray, rng: np.random.Generator):
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(M, axis=0)
    for r, nrm in enumerate(norms):
        if (not np.isfinite(nrm)) or nrm < 1e-12:
            M[:, r] = rng.normal(0.0, 1.0, size=M.shape[0])
            nrm = np.linalg.norm(M[:, r])
        M[:, r] /= nrm + 1e-12
    return M


def _design_for_A(B, C, D):
    J, R = B.shape
    K = C.shape[0]
    L = D.shape[0]
    Z = np.empty((J * K * L, R), dtype=float)
    for r in range(R):
        Z[:, r] = np.einsum("j,k,l->jkl", B[:, r], C[:, r], D[:, r]).ravel(order="C")
    return Z


def _design_for_B(A, C, D):
    I, R = A.shape
    K = C.shape[0]
    L = D.shape[0]
    Z = np.empty((I * K * L, R), dtype=float)
    for r in range(R):
        Z[:, r] = np.einsum("i,k,l->ikl", A[:, r], C[:, r], D[:, r]).ravel(order="C")
    return Z


def _design_for_C(A, B, D):
    I, R = A.shape
    J = B.shape[0]
    L = D.shape[0]
    Z = np.empty((I * J * L, R), dtype=float)
    for r in range(R):
        Z[:, r] = np.einsum("i,j,l->ijl", A[:, r], B[:, r], D[:, r]).ravel(order="C")
    return Z


def _design_for_D(A, B, C):
    I, R = A.shape
    J = B.shape[0]
    K = C.shape[0]
    Z = np.empty((I * J * K, R), dtype=float)
    for r in range(R):
        Z[:, r] = np.einsum("i,j,k->ijk", A[:, r], B[:, r], C[:, r]).ravel(order="C")
    return Z


@st.cache_data(ttl=7200, show_spinner=False)
def cpd_als_4d_factor_only(tensor: np.ndarray, rank: int, max_iter: int = 100, tol: float = 1e-6):
    """
    Same CPD-ALS logic as Tensor_Decomposition_temperature_scatter.py.
    This intentionally uses the original unregularized row-wise least-squares updates,
    the original SVD initialization for A, random initialization for B and C,
    and thermodynamic-prior initialization for D.
    """
    I, J, K, L = tensor.shape
    mask = ~np.isnan(tensor)
    X = np.where(mask, tensor, 0.0)

    if L == 31:
        T_vals_physical = np.array(list(range(700, 3701, 100)))
        T_mean, T_std = np.mean(T_vals_physical), np.std(T_vals_physical)
        T_norm = (T_vals_physical - T_mean) / (T_std + 1e-12)
        D = np.zeros((L, rank))
        D[:, 0] = 1.0
        if rank >= 2:
            D[:, 1] = T_norm
        if rank >= 3:
            D[:, 2] = (T_norm**2 - 1.0) * 0.5
        if rank >= 4:
            D[:, 3] = np.tanh(2.0 * T_norm) - np.mean(np.tanh(2.0 * T_norm))
        if rank > 4:
            D[:, 4:] = np.random.rand(L, rank - 4) * 0.01
    else:
        D = np.random.rand(L, rank) * 0.1

    X_unfolded = unfold_tensor(X, mode=0)
    try:
        U, svals, Vh = linalg.svd(X_unfolded, full_matrices=False)
        A = U[:, :rank] * np.sqrt(svals[:rank])
    except Exception:
        A = np.random.rand(I, rank) * 0.1

    B = np.random.rand(J, rank) * 0.1
    C = np.random.rand(K, rank) * 0.1

    prev_error = np.inf
    error = np.inf
    progress = st.empty()

    for iteration in range(max_iter):
        BCD = np.zeros((J * K * L, rank))
        for r in range(rank):
            BCD[:, r] = np.kron(np.kron(D[:, r], C[:, r]), B[:, r])
        X_flat = X.reshape(I, -1)
        mask_flat = mask.reshape(I, -1)
        for i in range(I):
            valid = mask_flat[i, :]
            if np.sum(valid) > rank:
                A[i, :] = linalg.lstsq(BCD[valid, :], X_flat[i, valid])[0]
        A = A / (np.linalg.norm(A, axis=0) + 1e-12)

        ACD = np.zeros((I * K * L, rank))
        for r in range(rank):
            ACD[:, r] = np.kron(np.kron(D[:, r], C[:, r]), A[:, r])
        X_flat = X.transpose(1, 0, 2, 3).reshape(J, -1)
        mask_flat = mask.transpose(1, 0, 2, 3).reshape(J, -1)
        for j in range(J):
            valid = mask_flat[j, :]
            if np.sum(valid) > rank:
                B[j, :] = linalg.lstsq(ACD[valid, :], X_flat[j, valid])[0]
        B = B / (np.linalg.norm(B, axis=0) + 1e-12)

        ABD = np.zeros((I * J * L, rank))
        for r in range(rank):
            ABD[:, r] = np.kron(np.kron(D[:, r], B[:, r]), A[:, r])
        X_flat = X.transpose(2, 0, 1, 3).reshape(K, -1)
        mask_flat = mask.transpose(2, 0, 1, 3).reshape(K, -1)
        for k in range(K):
            valid = mask_flat[k, :]
            if np.sum(valid) > rank:
                C[k, :] = linalg.lstsq(ABD[valid, :], X_flat[k, valid])[0]
        C = C / (np.linalg.norm(C, axis=0) + 1e-12)

        ABC = np.zeros((I * J * K, rank))
        for r in range(rank):
            ABC[:, r] = np.kron(np.kron(C[:, r], B[:, r]), A[:, r])
        X_flat = X.transpose(3, 0, 1, 2).reshape(L, -1)
        mask_flat = mask.transpose(3, 0, 1, 2).reshape(L, -1)
        for t in range(L):
            valid = mask_flat[t, :]
            if np.sum(valid) > rank:
                D[t, :] = linalg.lstsq(ABC[valid, :], X_flat[t, valid])[0]
        D = D / (np.linalg.norm(D, axis=0) + 1e-12)

        if iteration == 0 or (iteration + 1) % 5 == 0 or iteration == max_iter - 1:
            recon = np.zeros_like(X)
            for r in range(rank):
                recon += np.outer(A[:, r], np.kron(np.kron(D[:, r], C[:, r]), B[:, r])).reshape(I, J, K, L)
            residual = (tensor - recon)[mask]
            error = float(np.sqrt(np.mean(residual**2))) if len(residual) else np.inf
            progress.caption(f"ALS iteration {iteration+1}/{max_iter} | RMSE = {error:.6g}")
            if abs(prev_error - error) < tol:
                break
            prev_error = error

    progress.empty()

    lam = np.ones(rank)
    for r in range(rank):
        lam[r] = np.linalg.norm(A[:, r]) * np.linalg.norm(B[:, r]) * np.linalg.norm(C[:, r]) * np.linalg.norm(D[:, r])

    return A, B, C, D, lam, error



def factor_matrices_to_long_csv(A, B, C, D, lam, co_vals, cr_vals, fe_vals, T_vals, phase="LIQUID", R=6):
    """
    Create a long-format dataframe containing exactly the values plotted in the
    Unified CPD Factor Matrix View.

    For each CPD component r, the plotted y-values are:
      λ·A, λ·B, λ·C, and λ·D
    """
    co_arr = np.asarray(co_vals, dtype=float)
    cr_arr = np.asarray(cr_vals, dtype=float)
    fe_arr = np.asarray(fe_vals, dtype=float)
    T_arr = np.asarray(T_vals, dtype=float)
    R_eff = min(int(R), len(lam), A.shape[1], B.shape[1], C.shape[1], D.shape[1])

    blocks = [
        ("A", "Co", "x_Co", co_arr, A, "λ·A"),
        ("B", "Cr", "x_Cr", cr_arr, B, "λ·B"),
        ("C", "Fe", "x_Fe", fe_arr, C, "λ·C"),
        ("D", "Temperature", "Temperature_K", T_arr, D, "λ·D"),
    ]

    rows = []
    for matrix_name, variable, x_label, x_values, Fmat, y_label in blocks:
        for r in range(R_eff):
            factor_values = Fmat[:, r]
            plotted_values = lam[r] * factor_values
            for x, raw_y, plotted_y in zip(x_values, factor_values, plotted_values):
                rows.append({
                    "phase": phase,
                    "matrix": matrix_name,
                    "variable": variable,
                    "x_label": x_label,
                    "x_value": float(x),
                    "component": f"r={r+1}",
                    "component_index": r + 1,
                    "lambda": float(lam[r]),
                    "factor_value": float(raw_y),
                    "plotted_value": float(plotted_y),
                    "y_label": y_label,
                })

    return pd.DataFrame(rows)


def factor_matrices_to_wide_csv(long_df: pd.DataFrame):
    """Create a wide-format dataframe with one column per CPD component."""
    if long_df.empty:
        return long_df.copy()
    wide = long_df.pivot_table(
        index=["phase", "matrix", "variable", "x_label", "x_value", "y_label"],
        columns="component",
        values="plotted_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def plot_unified_factor_matrices(A, B, C, D, lam, co_vals, cr_vals, fe_vals, T_vals, phase="LIQUID", R=6):
    co_arr = np.asarray(co_vals, dtype=float)
    cr_arr = np.asarray(cr_vals, dtype=float)
    fe_arr = np.asarray(fe_vals, dtype=float)
    T_arr = np.asarray(T_vals, dtype=float)
    R_eff = min(int(R), len(lam), A.shape[1], B.shape[1], C.shape[1], D.shape[1])
    colors = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12", "#9b59b6", "#1abc9c", "#34495e", "#e67e22"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"A: Co factor (λ·A) — {phase}",
            f"B: Cr factor (λ·B) — {phase}",
            f"C: Fe factor (λ·C) — {phase}",
            f"D: Temperature factor (λ·D) — {phase}",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "scatter"}]],
    )

    for r in range(R_eff):
        color = colors[r % len(colors)]
        fig.add_trace(go.Scatter(x=co_arr, y=lam[r] * A[:, r], mode="lines+markers", name=f"r={r+1}", line=dict(color=color, width=2), marker=dict(size=5), legendgroup=f"r{r+1}", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=cr_arr, y=lam[r] * B[:, r], mode="lines+markers", name=f"r={r+1}", line=dict(color=color, width=2), marker=dict(size=5), legendgroup=f"r{r+1}", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=fe_arr, y=lam[r] * C[:, r], mode="lines+markers", name=f"r={r+1}", line=dict(color=color, width=2), marker=dict(size=5), legendgroup=f"r{r+1}", showlegend=False), row=2, col=1)
        fig.add_trace(
            go.Scatter(
                x=T_arr,
                y=lam[r] * D[:, r],
                mode="lines+markers",
                name=f"r={r+1}",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                legendgroup=f"r{r+1}",
                showlegend=False,
                hovertemplate="Temperature=%{x:.0f} K<br>λ·D=%{y:.6g}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="x_Co", row=1, col=1)
    fig.update_xaxes(title_text="x_Cr", row=1, col=2)
    fig.update_xaxes(title_text="x_Fe", row=2, col=1)
    fig.update_xaxes(title_text="Temperature (K)", row=2, col=2)
    fig.update_yaxes(title_text="λ·A", row=1, col=1)
    fig.update_yaxes(title_text="λ·B", row=1, col=2)
    fig.update_yaxes(title_text="λ·C", row=2, col=1)
    fig.update_yaxes(title_text="λ·D", row=2, col=2)
    fig.update_layout(
        height=900,
        title_text=f"Unified CPD Factor Matrices — {phase} Phase (R={R_eff})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title="Component"),
        template="plotly_white",
    )
    return fig


def run_cpd_pair(tdt_data, rank, max_iter):
    tensor_liq, _, _ = normalize_tensor(tdt_data["G_LIQ"])
    tensor_fcc, _, _ = normalize_tensor(tdt_data["G_FCC"])

    A_liq, B_liq, C_liq, D_liq, lam_liq, err_liq = cpd_als_4d_factor_only(tensor_liq, rank, max_iter=max_iter, tol=1e-5)
    A_fcc, B_fcc, C_fcc, D_fcc, lam_fcc, err_fcc = cpd_als_4d_factor_only(tensor_fcc, rank, max_iter=max_iter, tol=1e-5)

    st.session_state["factor_result"] = {
        "rank": rank,
        "A_liq": A_liq,
        "B_liq": B_liq,
        "C_liq": C_liq,
        "D_liq": D_liq,
        "lam_liq": lam_liq,
        "err_liq": err_liq,
        "A_fcc": A_fcc,
        "B_fcc": B_fcc,
        "C_fcc": C_fcc,
        "D_fcc": D_fcc,
        "lam_fcc": lam_fcc,
        "err_fcc": err_fcc,
    }


st.title("🔢 CPD Factor Matrices Calculation Only")
st.markdown(
    r"""
This standalone app calculates and displays only the **Unified CPD Factor Matrix View (All Four Matrices)**
for the **LIQUID** and **FCC** Gibbs-energy tensors.

\[
G(i,j,k,t) \approx \sum_{r=1}^{R} \lambda_r A_{ir}B_{jr}C_{kr}D_{tr}
\]
"""
)

with st.sidebar:
    st.header("Input")
    csv_dir = st.text_input("CSV folder", DEFAULT_CSV_DIR)
    rank = st.slider("CP rank R", 1, 12, 6, 1)
    max_iter = st.slider("ALS iterations", 10, 300, 150, 10)
    if st.button("Clear cached CPD result", use_container_width=True):
        st.session_state.pop("factor_result", None)
        st.rerun()


df = load_all_data(csv_dir)
tdt_data = build_tensor_data(df)
n_co, n_cr, n_fe, n_T = tdt_data["dims"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Co", n_co)
m2.metric("Cr", n_cr)
m3.metric("Fe", n_fe)
m4.metric("Temperature", n_T)
m5.metric("Valid entries", f"{tdt_data['valid_count']:,}")

if st.button("🚀 Run Factor Matrix CPD for LIQUID and FCC", type="primary", use_container_width=True):
    with st.spinner("Running factor-matrix CPD for LIQUID and FCC..."):
        run_cpd_pair(tdt_data, rank, max_iter)
    st.success("CPD factor matrices completed.")

if "factor_result" not in st.session_state:
    st.info("Click the run button to calculate the LIQUID and FCC factor matrices.")
    st.stop()

f = st.session_state["factor_result"]

if not np.isfinite(f["err_liq"]):
    st.warning("LIQUID RMSE is not finite. This matches the original unregularized CPD-ALS; try reducing rank or max iterations if needed.")
if not np.isfinite(f["err_fcc"]):
    st.warning("FCC RMSE is not finite. This matches the original unregularized CPD-ALS; try reducing rank or max iterations if needed.")

st.success(f"LIQUID RMSE = {f['err_liq']:.6g} | FCC RMSE = {f['err_fcc']:.6g}  (normalized tensor units; original ALS)")

fig_liq = plot_unified_factor_matrices(
    f["A_liq"], f["B_liq"], f["C_liq"], f["D_liq"], f["lam_liq"],
    tdt_data["co_vals"], tdt_data["cr_vals"], tdt_data["fe_vals"], tdt_data["T_vals"],
    phase="LIQUID", R=f["rank"],
)
fig_fcc = plot_unified_factor_matrices(
    f["A_fcc"], f["B_fcc"], f["C_fcc"], f["D_fcc"], f["lam_fcc"],
    tdt_data["co_vals"], tdt_data["cr_vals"], tdt_data["fe_vals"], tdt_data["T_vals"],
    phase="FCC", R=f["rank"],
)


liq_csv_df = factor_matrices_to_long_csv(
    f["A_liq"], f["B_liq"], f["C_liq"], f["D_liq"], f["lam_liq"],
    tdt_data["co_vals"], tdt_data["cr_vals"], tdt_data["fe_vals"], tdt_data["T_vals"],
    phase="LIQUID", R=f["rank"],
)
fcc_csv_df = factor_matrices_to_long_csv(
    f["A_fcc"], f["B_fcc"], f["C_fcc"], f["D_fcc"], f["lam_fcc"],
    tdt_data["co_vals"], tdt_data["cr_vals"], tdt_data["fe_vals"], tdt_data["T_vals"],
    phase="FCC", R=f["rank"],
)
combined_csv_df = pd.concat([liq_csv_df, fcc_csv_df], ignore_index=True)
liq_wide_df = factor_matrices_to_wide_csv(liq_csv_df)
fcc_wide_df = factor_matrices_to_wide_csv(fcc_csv_df)
combined_wide_df = factor_matrices_to_wide_csv(combined_csv_df)

with st.expander("⬇️ Download plotted factor-matrix data as CSV", expanded=True):
    st.caption(
        "The column `plotted_value` is the exact y-value used in the plots: "
        "λ·A, λ·B, λ·C, or λ·D. Use the long CSV for Python/Plotly/Origin; "
        "use the wide CSV if you want one column per component."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download LIQUID long CSV",
            liq_csv_df.to_csv(index=False).encode("utf-8"),
            file_name=f"LIQUID_factor_matrices_R{f['rank']}_long.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download LIQUID wide CSV",
            liq_wide_df.to_csv(index=False).encode("utf-8"),
            file_name=f"LIQUID_factor_matrices_R{f['rank']}_wide.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download FCC long CSV",
            fcc_csv_df.to_csv(index=False).encode("utf-8"),
            file_name=f"FCC_factor_matrices_R{f['rank']}_long.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download FCC wide CSV",
            fcc_wide_df.to_csv(index=False).encode("utf-8"),
            file_name=f"FCC_factor_matrices_R{f['rank']}_wide.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "Download LIQUID + FCC long CSV",
            combined_csv_df.to_csv(index=False).encode("utf-8"),
            file_name=f"LIQUID_FCC_factor_matrices_R{f['rank']}_long.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download LIQUID + FCC wide CSV",
            combined_wide_df.to_csv(index=False).encode("utf-8"),
            file_name=f"LIQUID_FCC_factor_matrices_R{f['rank']}_wide.csv",
            mime="text/csv",
            use_container_width=True,
        )

tab_liq, tab_fcc, tab_side = st.tabs(["LIQUID", "FCC", "Side-by-side"])

with tab_liq:
    st.plotly_chart(fig_liq, use_container_width=True, key="liq_factor_matrix")

with tab_fcc:
    st.plotly_chart(fig_fcc, use_container_width=True, key="fcc_factor_matrix")

with tab_side:
    col_liq, col_fcc = st.columns(2)
    with col_liq:
        st.subheader("LIQUID")
        st.plotly_chart(fig_liq, use_container_width=True, key="liq_factor_matrix_side")
    with col_fcc:
        st.subheader("FCC")
        st.plotly_chart(fig_fcc, use_container_width=True, key="fcc_factor_matrix_side")
