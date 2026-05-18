# import streamlit as st
# import numpy as np
# import torch
# import torch.nn as nn
# import pandas as pd
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# import re
# import os

# # === Initialize Session State ===
# if 'interpolated_T' not in st.session_state:
#     st.session_state.interpolated_T = None
# if 'results' not in st.session_state:
#     st.session_state.results = None
# if 'shape' not in st.session_state:
#     st.session_state.shape = None
# if 'sources' not in st.session_state:
#     st.session_state.sources = None
# if 'computed_params' not in st.session_state:
#     st.session_state.computed_params = None

# # === Page Config ===
# st.set_page_config(page_title="Attention-Based Laser Temperature Interpolation", layout="wide")
# st.title("Attention-Driven Interpolation of Laser-Induced Temperature Fields")
# st.markdown("""
# **Scientific Context**: This tool uses **transformer-inspired cross-attention with Gaussian spatial locality regularization** to interpolate 3D spatiotemporal temperature fields \( T(t, y, x) \) from precomputed FEM simulations.

# Parameters \( P \) (W) and \( v \) (m/s) are parsed from filenames.

# **Visualization**: Assuming y-index 0 = bottom and high y-index = top surface (laser heating), displays are oriented with **hot region at top** and **equal aspect ratio** (square cells assuming uniform grid spacing).
# """)

# # === Model Definition ===
# class MultiParamAttentionInterpolator(nn.Module):
#     def __init__(self, sigma=0.2, num_heads=4, d_head=8):
#         super().__init__()
#         self.sigma = sigma
#         self.num_heads = num_heads
#         self.d_head = d_head
#         self.W_q = nn.Linear(2, num_heads * d_head, bias=False)
#         self.W_k = nn.Linear(2, num_heads * d_head, bias=False)

#     def normalize_params(self, params_list, target_params):
#         all_params = np.array(params_list + [target_params])
#         mins = all_params.min(axis=0)
#         maxs = all_params.max(axis=0)
#         range_ = maxs - mins + 1e-8
#         norm_sources = (np.array(params_list) - mins) / range_
#         norm_target = (np.array(target_params) - mins) / range_
#         return torch.tensor(norm_sources, dtype=torch.float32), torch.tensor(norm_target, dtype=torch.float32)

#     def compute_weights(self, params_list, p_target, v_target):
#         target_params = [p_target, v_target]
#         src_tensor, tgt_tensor_1d = self.normalize_params(params_list, target_params)
#         tgt_tensor = tgt_tensor_1d.unsqueeze(0)

#         q = self.W_q(tgt_tensor).view(1, self.num_heads, self.d_head)
#         k = self.W_k(src_tensor).view(len(params_list), self.num_heads, self.d_head)

#         attn_logits = torch.einsum('nhd,mhd->nmh', k, q) / np.sqrt(self.d_head)
#         attn_weights = torch.softmax(attn_logits.squeeze(1), dim=0)
#         attn_weights = attn_weights.mean(dim=1)

#         diffs = src_tensor - tgt_tensor
#         dists = torch.sqrt(torch.sum(diffs**2, dim=1))
#         spatial_weights = torch.exp(-dists**2 / (2 * self.sigma**2))
#         spatial_weights = spatial_weights / (spatial_weights.sum() + 1e-8)

#         combined = attn_weights * spatial_weights
#         combined = combined / (combined.sum() + 1e-8)

#         return {
#             'attention_weights': attn_weights.detach().numpy(),
#             'spatial_weights': spatial_weights.detach().numpy(),
#             'combined_weights': combined.detach().numpy(),
#             'norm_sources': src_tensor.numpy(),
#             'norm_target': tgt_tensor_1d.numpy(),
#             'W_q': self.W_q.weight.data.numpy(),
#             'W_k': self.W_k.weight.data.numpy()
#         }

# # === Sidebar: Controls ===
# with st.sidebar:
#     st.header("Attention Model")
#     sigma = st.slider("Gaussian Locality σ", 0.05, 0.50, 0.20, 0.01)
#     num_heads = st.slider("Attention Heads", 1, 8, 4)
#     d_head = st.slider("Dim per Head", 4, 16, 8)
#     seed = st.number_input("Random Seed", 0, 9999, 42)
#     torch.manual_seed(seed)
#     np.random.seed(seed)

#     st.header("Target Parameters")
#     p_target = st.number_input("Target Laser Power \( P^* \) (W)", 100.0, 1000.0, 350.0, 10.0)
#     v_target = st.number_input("Target Scan Velocity \( v^* \) (m/s)", 0.1, 3.0, 0.5, 0.05)

# # === Load Method Selection ===
# load_method = st.selectbox("Source Loading Method", ["Upload files", "From local folder"])

# sources = []
# loaded_arrays = []
# param_list = []

# if load_method == "Upload files":
#     uploaded_files = st.file_uploader(
#         "Upload multiple .npy files containing 3D T(t,y,x)",
#         type=["npy"],
#         accept_multiple_files=True
#     )
#     items_to_process = uploaded_files or []
#     def get_filename(item): return item.name
#     def load_array(item): 
#         item.seek(0)
#         return np.load(item)
# else:
#     folder_path = st.text_input("Local folder path (e.g., 't_folder')", "t_folder")
#     if folder_path and os.path.isdir(folder_path):
#         npy_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.npy')])
#         st.info(f"Found {len(npy_paths)} .npy files in '{folder_path}'")
#         items_to_process = npy_paths
#         def get_filename(path): return os.path.basename(path)
#         def load_array(path): return np.load(path)
#     else:
#         items_to_process = []
#         if folder_path:
#             st.warning(f"Folder '{folder_path}' not found or invalid.")

# # === Process Sources ===
# if items_to_process:
#     for item in items_to_process:
#         filename = get_filename(item).lower()
#         match = re.search(r'(\d+)w(\d+)v\.npy$', filename, re.IGNORECASE)
#         if not match:
#             st.warning(f"Skipping '{get_filename(item)}': filename must end with <Power>W<VelocityDigits>V.npy")
#             continue

#         p_str, v_digits = match.groups()
#         p = int(p_str)
#         if len(v_digits) == 0:
#             st.warning(f"Skipping '{get_filename(item)}': no velocity digits")
#             continue
#         v = int(v_digits) / (10 ** (len(v_digits) - 1))

#         try:
#             T_arr = load_array(item)
#             if T_arr.ndim != 3:
#                 st.error(f"'{get_filename(item)}': Expected 3D array (Nt, Ny, Nx), got {T_arr.shape}")
#                 continue

#             sources.append({
#                 'file_name': get_filename(item),
#                 'P': p,
#                 'v': v,
#                 'shape': T_arr.shape
#             })
#             loaded_arrays.append(T_arr)
#             param_list.append((p, v))

#             st.success(f"Loaded '{get_filename(item)}' → P = {p} W, v = {v:.3f} m/s, shape {T_arr.shape}")
#         except Exception as e:
#             st.error(f"Error loading '{get_filename(item)}': {e}")

# if sources:
#     st.subheader(f"Loaded {len(sources)} Valid Source Simulations")
#     df_sources = pd.DataFrame(sources)[['file_name', 'P', 'v', 'shape']]
#     st.dataframe(df_sources.style.format({'v': '{:.3f}'}), use_container_width=True)

#     shapes = [s['shape'] for s in sources]
#     if len(set(shapes)) != 1:
#         st.error("❌ All source arrays must have identical shape (Nt × Ny × Nx)")
#         loaded_arrays = []
#     else:
#         st.success(f"✅ All sources consistent: {shapes[0]}")

# # === Interpolation Button ===
# current_params = (p_target, v_target, sigma, num_heads, d_head, tuple(param_list))
# if st.button("Run Attention-Based Interpolation", type="primary") or \
#    (st.session_state.interpolated_T is not None and st.session_state.computed_params != current_params):
    
#     if len(loaded_arrays) < 2:
#         st.error("Need at least 2 valid sources")
#     else:
#         with st.spinner("Computing hybrid weights and interpolating..."):
#             shape = loaded_arrays[0].shape
#             Nt, Ny, Nx = shape

#             interpolator = MultiParamAttentionInterpolator(sigma=sigma, num_heads=num_heads, d_head=d_head)
#             results = interpolator.compute_weights(param_list, p_target, v_target)

#             weights = results['combined_weights']

#             interpolated_T = np.zeros(shape, dtype=np.float64)
#             for w, arr in zip(weights, loaded_arrays):
#                 interpolated_T += w * arr

#             st.session_state.interpolated_T = interpolated_T
#             st.session_state.results = results
#             st.session_state.shape = shape
#             st.session_state.sources = sources
#             st.session_state.computed_params = current_params

#         st.success("Interpolation complete and cached!")

# # === Display Results ===
# if st.session_state.interpolated_T is not None:
#     interpolated_T = st.session_state.interpolated_T
#     results = st.session_state.results
#     shape = st.session_state.shape
#     sources = st.session_state.sources
#     Nt, Ny, Nx = shape

#     col1, col2 = st.columns([1.3, 1])
#     with col1:
#         st.subheader("Hybrid Attention Weights")
#         df_weights = pd.DataFrame({
#             'Source File': [s['file_name'] for s in sources],
#             'P (W)': [s['P'] for s in sources],
#             'v (m/s)': [f"{s['v']:.3f}" for s in sources],
#             'Attention': np.round(results['attention_weights'], 4),
#             'Gaussian': np.round(results['spatial_weights'], 4),
#             'Hybrid': np.round(results['combined_weights'], 4)
#         })
#         st.dataframe(df_weights.style.bar(subset=['Hybrid'], color='#5fba7d'), use_container_width=True)

#         fig_param = go.Figure()
#         src_norm = results['norm_sources']
#         tgt_norm = results['norm_target']
#         fig_param.add_trace(go.Scatter(
#             x=src_norm[:,0], y=src_norm[:,1],
#             mode='markers+text',
#             marker=dict(size=20, color=results['combined_weights'], colorscale='Viridis', showscale=True),
#             text=[f"P={s['P']}W<br>v={s['v']:.3f}m/s" for s in sources],
#             textposition="top center",
#             name='Sources'
#         ))
#         fig_param.add_trace(go.Scatter(
#             x=[tgt_norm[0]], y=[tgt_norm[1]],
#             mode='markers+text',
#             marker=dict(size=30, symbol='star', color='red'),
#             text=f"Target<br>P={p_target}W<br>v={v_target:.3f}m/s",
#             textposition="bottom center",
#             name='Target'
#         ))
#         fig_param.update_layout(
#             title="Parameter Space (normalized P vs v)",
#             xaxis_title="Normalized Power",
#             yaxis_title="Normalized Velocity"
#         )
#         st.plotly_chart(fig_param, use_container_width=True)

#     with col2:
#         st.subheader("Learned Projection Matrices")
#         fig_proj, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
#         sns.heatmap(results['W_q'], ax=ax1, cmap='coolwarm', center=0, cbar=False)
#         ax1.set_title("$W_q$ (Query)")
#         sns.heatmap(results['W_k'], ax=ax2, cmap='coolwarm', center=0)
#         ax2.set_title("$W_k$ (Key)")
#         st.pyplot(fig_proj)

#     st.subheader(f"Interpolated Temperature Field @ (P* = {p_target} W, v* = {v_target:.3f} m/s)")
#     st.info(f"Field shape: {shape} → {Nt} time steps × {Ny} y-points × {Nx} x-points")

#     # Animated heatmap (Plotly: flipud to put high y-index at top)
#     frames = []
#     t_min, t_max = interpolated_T.min(), interpolated_T.max()
#     for k in range(Nt):
#         display_slice = np.flipud(interpolated_T[k, :, :])  # High y-index at top
#         frames.append(go.Frame(
#             data=go.Heatmap(z=display_slice, colorscale='Inferno', zmin=t_min, zmax=t_max),
#             name=f"t{k}"
#         ))

#     fig = go.Figure(data=frames[0].data, frames=frames)
#     fig.update_layout(
#         title="Temperature Evolution (play or slide)",
#         xaxis_title="X index",
#         yaxis_title="Y index (top = surface)",
#         updatemenus=[{
#             "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 150, "redraw": True}}]}],
#             "direction": "left", "pad": {"r": 10, "t": 87}, "type": "buttons", "x": 0.1, "y": 0
#         }]
#     )
#     sliders = [{"active": 0, "steps": [{"method": "animate", "args": [[f"t{k}"], {"mode": "immediate"}], "label": f"{k}"} for k in range(Nt)],
#                 "x": 0.1, "len": 0.9, "y": 0}]
#     fig.update_layout(sliders=sliders)
#     # Equal aspect ratio
#     fig.update_yaxes(scaleanchor="x", scaleratio=1)
#     st.plotly_chart(fig, use_container_width=True)

#     # Static contour (matplotlib: origin='lower' for high y-index at top)
#     with st.expander("Static Contour Plots"):
#         t_idx = st.slider("Time index", 0, Nt-1, Nt//2, key="static_t")
#         slice_original = interpolated_T[t_idx, :, :]  # No flip
#         fig_static, ax = plt.subplots(figsize=(10, 8))
#         cont = ax.contourf(slice_original, levels=60, cmap='jet', origin='upper')
#         ax.set_aspect('equal')  # Realistic square cells
#         ax.set_title(f"Temperature @ t = {t_idx} (laser heating at top)")
#         ax.set_xlabel("X index")
#         ax.set_ylabel("Y index (top = surface)")
#         plt.colorbar(cont, ax=ax, label="Temperature")
#         st.pyplot(fig_static)

#     # === Downloads ===
#     st.subheader("Download Interpolated Field")
#     col_d1, col_d2 = st.columns(2)
#     with col_d1:
#         buffer_npz = io.BytesIO()
#         np.savez_compressed(buffer_npz, T=interpolated_T, P=p_target, v=v_target)
#         buffer_npz.seek(0)
#         st.download_button(
#             "Download .npz (original orientation)",
#             buffer_npz,
#             file_name=f"T_interp_P{p_target}W_v{v_target:.3f}ms.npz",
#             mime="application/octet-stream"
#         )
#     with col_d2:
#         flat_df = pd.DataFrame({
#             't_idx': np.repeat(np.arange(Nt), Ny * Nx),
#             'y_idx': np.tile(np.repeat(np.arange(Ny), Nx), Nt),
#             'x_idx': np.tile(np.arange(Nx), Ny * Nt),
#             'Temperature': interpolated_T.flatten()
#         })
#         csv_buf = io.StringIO()
#         flat_df.to_csv(csv_buf, index=False)
#         st.download_button(
#             "Download .csv (flattened, original orientation)",
#             csv_buf.getvalue(),
#             file_name=f"T_interp_P{p_target}W_v{v_target:.3f}ms.csv",
#             mime="text/csv"
#         )

# else:
#     st.info("Load sources and click 'Run Attention-Based Interpolation' to compute.")










import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
import os

TIME45 = np.arange(30.5, 1648.5+2, 2)[0::33]
TIME50 = np.arange(30.5, 1518.5+2, 2)[0::30]
TIME60 = np.arange(30.5, 1268.5+2, 2)[0::25]
TIME70 = np.arange(30.5, 1088.5+2, 2)[0::22]

# === Initialize Session State ===
if 'interpolated_T' not in st.session_state:
    st.session_state.interpolated_T = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'shape' not in st.session_state:
    st.session_state.shape = None
if 'sources' not in st.session_state:
    st.session_state.sources = None
if 'computed_params' not in st.session_state:
    st.session_state.computed_params = None

# === Page Config ===
st.set_page_config(page_title="Attention-Based Laser Temperature Interpolation", layout="wide")
st.title("Attention-Driven Interpolation of Laser-Induced Temperature Fields")
st.markdown("""
**Scientific Context**: This tool uses **transformer-inspired cross-attention with Gaussian spatial locality regularization** to interpolate 3D spatiotemporal temperature fields \( T(t, y, x) \) from precomputed FEM simulations.

Parameters \( P \) (W) and \( v \) (m/s) are parsed from filenames.

**Visualization**: Assuming y-index 0 = bottom and high y-index = top surface (laser heating), displays are oriented with **hot region at top** and **equal aspect ratio** (square cells assuming uniform grid spacing).
""")

# === Model Definition ===
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(2, num_heads * d_head, bias=False)
        self.W_k = nn.Linear(2, num_heads * d_head, bias=False)

    def normalize_params(self, params_list, target_params):
        all_params = np.array(params_list + [target_params])
        mins = all_params.min(axis=0)
        maxs = all_params.max(axis=0)
        range_ = maxs - mins + 1e-8
        norm_sources = (np.array(params_list) - mins) / range_
        norm_target = (np.array(target_params) - mins) / range_
        return torch.tensor(norm_sources, dtype=torch.float32), torch.tensor(norm_target, dtype=torch.float32)

    def compute_weights(self, params_list, p_target, v_target):
        target_params = [p_target, v_target]
        src_tensor, tgt_tensor_1d = self.normalize_params(params_list, target_params)
        tgt_tensor = tgt_tensor_1d.unsqueeze(0)

        q = self.W_q(tgt_tensor).view(1, self.num_heads, self.d_head)
        k = self.W_k(src_tensor).view(len(params_list), self.num_heads, self.d_head)

        attn_logits = torch.einsum('nhd,mhd->nmh', k, q) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits.squeeze(1), dim=0)
        attn_weights = attn_weights.mean(dim=1)

        diffs = src_tensor - tgt_tensor
        dists = torch.sqrt(torch.sum(diffs**2, dim=1))
        spatial_weights = torch.exp(-dists**2 / (2 * self.sigma**2))
        spatial_weights = spatial_weights / (spatial_weights.sum() + 1e-8)

        combined = attn_weights * spatial_weights
        combined = combined / (combined.sum() + 1e-8)

        return {
            'attention_weights': attn_weights.detach().numpy(),
            'spatial_weights': spatial_weights.detach().numpy(),
            'combined_weights': combined.detach().numpy(),
            'norm_sources': src_tensor.numpy(),
            'norm_target': tgt_tensor_1d.numpy(),
            'W_q': self.W_q.weight.data.numpy(),
            'W_k': self.W_k.weight.data.numpy()
        }

# === Sidebar: Controls ===
with st.sidebar:
    st.header("Attention Model")
    sigma = st.slider("Gaussian Locality σ", 0.05, 0.50, 0.20, 0.01)
    num_heads = st.slider("Attention Heads", 1, 8, 4)
    d_head = st.slider("Dim per Head", 4, 16, 8)
    seed = st.number_input("Random Seed", 0, 9999, 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    st.header("Target Parameters")
    p_target = st.number_input("Target Laser Power \( P^* \) (W)", 300.0, 800.0, 420.0, 10.0)
    v_target = st.number_input("Target Scan Velocity \( v^* \) (m/s)", 0.4, 0.8, 0.45, 0.05)

# === Load Method Selection ===
load_method = st.selectbox("Source Loading Method", ["Upload files", "From local folder"])

sources = []
loaded_arrays = []
param_list = []

if load_method == "Upload files":
    uploaded_files = st.file_uploader(
        "Upload multiple .npy files containing 3D T(t,y,x)",
        type=["npy"],
        accept_multiple_files=True
    )
    items_to_process = uploaded_files or []
    def get_filename(item): return item.name
    def load_array(item): 
        item.seek(0)
        return np.load(item)
else:
    folder_path = st.text_input("Local folder path (e.g., 'TEMP')", "TEMP")
    if folder_path and os.path.isdir(folder_path):
        npy_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.npy')])
        st.info(f"Found {len(npy_paths)} .npy files in '{folder_path}'")
        items_to_process = npy_paths
        def get_filename(path): return os.path.basename(path)
        def load_array(path): return np.load(path)
    else:
        items_to_process = []
        if folder_path:
            st.warning(f"Folder '{folder_path}' not found or invalid.")

# === Process Sources ===
if items_to_process:
    for item in items_to_process:
        filename = get_filename(item).lower()
        match = re.search(r'p(\d+)s(\d+)\.npy$', filename)
        if not match:
            st.warning(f"Skipping '{get_filename(item)}': filename must end with p<Power>s<Speed>.npy (e.g., p350s45.npy)")
            continue

        p_str, v_str = match.groups()
        p = int(p_str)
        v = int(v_str) / 100.0  # Interpret speed digits as cm/s → convert to m/s

        try:
            T_arr = load_array(item)
            if T_arr.ndim != 3:
                st.error(f"'{get_filename(item)}': Expected 3D array (Nt, Ny, Nx), got {T_arr.shape}")
                continue

            sources.append({
                'file_name': get_filename(item),
                'P': p,
                'v': v,
                'shape': T_arr.shape
            })
            loaded_arrays.append(T_arr)
            param_list.append((p, v))

            st.success(f"Loaded '{get_filename(item)}' → P = {p} W, v = {v:.3f} m/s, shape {T_arr.shape}")
        except Exception as e:
            st.error(f"Error loading '{get_filename(item)}': {e}")

if sources:
    st.subheader(f"Loaded {len(sources)} Valid Source Simulations")
    df_sources = pd.DataFrame(sources)[['file_name', 'P', 'v', 'shape']]
    st.dataframe(df_sources.style.format({'v': '{:.3f}'}), use_container_width=True)

    shapes = [s['shape'] for s in sources]
    if len(set(shapes)) != 1:
        st.error("❌ All source arrays must have identical shape (Nt × Ny × Nx)")
        loaded_arrays = []
    else:
        st.success(f"✅ All sources consistent: {shapes[0]}")

# === Interpolation Button ===
current_params = (p_target, v_target, sigma, num_heads, d_head, tuple(param_list))
if st.button("Run Attention-Based Interpolation", type="primary") or \
   (st.session_state.interpolated_T is not None and st.session_state.computed_params != current_params):
    
    if len(loaded_arrays) < 2:
        st.error("Need at least 2 valid sources")
    else:
        with st.spinner("Computing hybrid weights and interpolating..."):
            shape = loaded_arrays[0].shape
            Nt, Ny, Nx = shape

            interpolator = MultiParamAttentionInterpolator(sigma=sigma, num_heads=num_heads, d_head=d_head)
            results = interpolator.compute_weights(param_list, p_target, v_target)

            weights = results['combined_weights']

            interpolated_T = np.zeros(shape, dtype=np.float64)
            for w, arr in zip(weights, loaded_arrays):
                interpolated_T += w * arr

            st.session_state.interpolated_T = interpolated_T
            st.session_state.results = results
            st.session_state.shape = shape
            st.session_state.sources = sources
            st.session_state.computed_params = current_params

        st.success("Interpolation complete and cached!")

# === Display Results ===
if st.session_state.interpolated_T is not None:
    interpolated_T = st.session_state.interpolated_T
    results = st.session_state.results
    shape = st.session_state.shape
    sources = st.session_state.sources
    Nt, Ny, Nx = shape

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.subheader("Hybrid Attention Weights")
        df_weights = pd.DataFrame({
            'Source File': [s['file_name'] for s in sources],
            'P (W)': [s['P'] for s in sources],
            'v (m/s)': [f"{s['v']:.3f}" for s in sources],
            'Attention': np.round(results['attention_weights'], 4),
            'Gaussian': np.round(results['spatial_weights'], 4),
            'Hybrid': np.round(results['combined_weights'], 4)
        })
        st.dataframe(df_weights.style.bar(subset=['Hybrid'], color='#5fba7d'), use_container_width=True)

        fig_param = go.Figure()
        src_norm = results['norm_sources']
        tgt_norm = results['norm_target']
        fig_param.add_trace(go.Scatter(
            x=src_norm[:,0], y=src_norm[:,1],
            mode='markers+text',
            marker=dict(size=20, color=results['combined_weights'], colorscale='Viridis', showscale=True),
            text=[f"P={s['P']}W<br>v={s['v']:.3f}m/s" for s in sources],
            textposition="top center",
            name='Sources'
        ))
        fig_param.add_trace(go.Scatter(
            x=[tgt_norm[0]], y=[tgt_norm[1]],
            mode='markers+text',
            marker=dict(size=30, symbol='star', color='red'),
            text=f"Target<br>P={p_target}W<br>v={v_target:.3f}m/s",
            textposition="bottom center",
            name='Target'
        ))
        fig_param.update_layout(
            title="Parameter Space (normalized P vs v)",
            xaxis_title="Normalized Power",
            yaxis_title="Normalized Velocity"
        )
        st.plotly_chart(fig_param, use_container_width=True)

    with col2:
        st.subheader("Learned Projection Matrices")
        fig_proj, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        sns.heatmap(results['W_q'], ax=ax1, cmap='coolwarm', center=0, cbar=False)
        ax1.set_title("$W_q$ (Query)")
        sns.heatmap(results['W_k'], ax=ax2, cmap='coolwarm', center=0)
        ax2.set_title("$W_k$ (Key)")
        st.pyplot(fig_proj)

    st.subheader(f"Interpolated Temperature Field @ (P* = {p_target} W, v* = {v_target:.3f} m/s)")
    st.info(f"Field shape: {shape} → {Nt} time steps × {Ny} y-points × {Nx} x-points")

    # Animated heatmap (Plotly: flipud to put high y-index at top)
    frames = []
    t_min, t_max = interpolated_T.min(), interpolated_T.max()
    for k in range(Nt):
        display_slice = np.flipud(interpolated_T[k, :, :])  # High y-index at top
        frames.append(go.Frame(
            data=go.Heatmap(z=display_slice, colorscale='Inferno', zmin=t_min, zmax=t_max),
            name=f"t{k}"
        ))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Temperature Evolution (play or slide)",
        xaxis_title="X index",
        yaxis_title="Y index (top = surface)",
        updatemenus=[{
            "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 150, "redraw": True}}]}],
            "direction": "left", "pad": {"r": 10, "t": 87}, "type": "buttons", "x": 0.1, "y": 0
        }]
    )
    sliders = [{"active": 0, "steps": [{"method": "animate", "args": [[f"t{k}"], {"mode": "immediate"}], "label": f"{k}"} for k in range(Nt)],
                "x": 0.1, "len": 0.9, "y": 0}]
    fig.update_layout(sliders=sliders)
    # Equal aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

    # Static contour (matplotlib: origin='lower' for high y-index at top)
    with st.expander("Static Contour Plots"):
        t_idx = st.slider("Time index", 0, Nt-1, Nt//2, key="static_t")
        # laser_pos = 125+((TIME45[t_idx]-3)*v_target)
        slice_original = interpolated_T[t_idx, :, :]  # No flip
        # st.write(f"Temperature @ time: {TIME45[t_idx]:.0f}μs (T-max ={slice_original.max():.0f}K)")

        fig_static, ax = plt.subplots(figsize=(10, 8))
        # ax.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
        hmap1= ax.imshow(slice_original, cmap='jet', vmin=300, aspect=1.0,  interpolation='quadric')
        ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        st.write(f"Temperature @ t-step: {t_idx} (T-max ={slice_original.max():.0f}K)")
        ax.set_ylim(260, -1);  ax.set_xlim(-1,1000)
        ax1a = fig_static.add_axes([0.92, 0.37, 0.04, 0.25]) 
        cbar = fig_static.colorbar(hmap1, cax=ax1a)
        st.pyplot(fig_static)

    # === Downloads ===
    st.subheader("Download Interpolated Field")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        buffer_npz = io.BytesIO()
        np.savez_compressed(buffer_npz, T=interpolated_T, P=p_target, v=v_target)
        buffer_npz.seek(0)
        st.download_button(
            "Download .npz (original orientation)",
            buffer_npz,
            file_name=f"T_interp_P{p_target}W_v{v_target:.3f}ms.npz",
            mime="application/octet-stream"
        )
    with col_d2:
        flat_df = pd.DataFrame({
            't_idx': np.repeat(np.arange(Nt), Ny * Nx),
            'y_idx': np.tile(np.repeat(np.arange(Ny), Nx), Nt),
            'x_idx': np.tile(np.arange(Nx), Ny * Nt),
            'Temperature': interpolated_T.flatten()
        })
        csv_buf = io.StringIO()
        flat_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download .csv (flattened, original orientation)",
            csv_buf.getvalue(),
            file_name=f"T_interp_P{p_target}W_v{v_target:.3f}ms.csv",
            mime="text/csv"
        )

else:
    st.info("Load sources and click 'Run Attention-Based Interpolation' to compute.")



