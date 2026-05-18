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
# if 'interpolated_eta' not in st.session_state:
#     st.session_state.interpolated_eta = None
# if 'results' not in st.session_state:
#     st.session_state.results = None
# if 'shape' not in st.session_state:
#     st.session_state.shape = None
# if 'sources' not in st.session_state:
#     st.session_state.sources = None
# if 'computed_params' not in st.session_state:
#     st.session_state.computed_params = None

# # === Page Config ===
# st.set_page_config(page_title="Attention-Based Multi-Field Interpolation", layout="wide")
# st.title("Attention-Driven Interpolation of Temperature & Phase Field")
# st.markdown("""
# **Scientific Context**: This tool interpolates **temperature \( T(t, y, x) \)** and **phase field \( \eta(t, y, x) \)** from paired FEM simulations.

# - Temperature files: `Tsol<P>W<V>V.npy` in `t_folder`
# - Phase field files: `ETAsol<P>W<V>V.npy` in `eta_folder`

# Both fields use the **same attention weights** from (P, v).
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

# # === Folders ===
# t_folder = st.text_input("Temperature folder path", "t_folder")
# eta_folder = st.text_input("Phase field folder path", "eta_folder")

# sources = []
# loaded_T_arrays = []
# loaded_eta_arrays = []
# param_list = []

# if os.path.isdir(t_folder):
#     npy_paths = sorted([os.path.join(t_folder, f) for f in os.listdir(t_folder) if f.lower().endswith('.npy')])
#     st.info(f"Found {len(npy_paths)} temperature files")

#     for path in npy_paths:
#         filename = os.path.basename(path).lower()
#         match = re.search(r'tsol(\d+)w(\d+)v\.npy$', filename)
#         if not match:
#             st.warning(f"Skipping {filename}")
#             continue

#         p_str, v_digits = match.groups()
#         p = int(p_str)
#         v = int(v_digits) / (10 ** (len(v_digits) - 1))

#         try:
#             T_arr = np.load(path)
#             if T_arr.ndim != 3:
#                 st.error(f"{filename}: Expected 3D")
#                 continue

#             eta_arr = None
#             if os.path.isdir(eta_folder):
#                 eta_path = os.path.join(eta_folder, f"ETAsol{p_str}W{v_digits}V.npy")
#                 if os.path.exists(eta_path):
#                     eta_arr = np.load(eta_path)
#                     if eta_arr.shape != T_arr.shape:
#                         st.warning(f"ETA shape mismatch for {filename}")
#                         eta_arr = None

#             sources.append({'file_name': os.path.basename(path), 'P': p, 'v': v, 'shape': T_arr.shape, 'eta': eta_arr is not None})
#             loaded_T_arrays.append(T_arr)
#             loaded_eta_arrays.append(eta_arr)
#             param_list.append((p, v))

#         except Exception as e:
#             st.error(f"Error {filename}: {e}")
# else:
#     st.error(f"Temperature folder '{t_folder}' not found")

# if sources:
#     df_sources = pd.DataFrame(sources)[['file_name', 'P', 'v', 'eta', 'shape']]
#     st.dataframe(df_sources.style.format({'v': '{:.3f}'}))

#     if len(set(s['shape'] for s in sources)) != 1:
#         st.error("Shapes must match")
#         loaded_T_arrays = []

# # === Interpolation ===
# current_params = (p_target, v_target, sigma, num_heads, d_head, tuple(param_list))
# if st.button("Run Interpolation", type="primary") or (st.session_state.interpolated_T is not None and st.session_state.computed_params != current_params):
#     if len(loaded_T_arrays) < 2:
#         st.error("Need ≥2 sources")
#     else:
#         with st.spinner("Interpolating..."):
#             shape = loaded_T_arrays[0].shape
#             Nt, Ny, Nx = shape

#             interpolator = MultiParamAttentionInterpolator(sigma=sigma, num_heads=num_heads, d_head=d_head)
#             results = interpolator.compute_weights(param_list, p_target, v_target)
#             weights = results['combined_weights']

#             interpolated_T = np.sum([w * arr for w, arr in zip(weights, loaded_T_arrays)], axis=0)

#             st.session_state.interpolated_T = interpolated_T
#             st.session_state.results = results
#             st.session_state.shape = shape
#             st.session_state.sources = sources
#             st.session_state.computed_params = current_params

#             if all(e is not None for e in loaded_eta_arrays):
#                 interpolated_eta = np.sum([w * arr for w, arr in zip(weights, loaded_eta_arrays)], axis=0)
#                 st.session_state.interpolated_eta = interpolated_eta
#             else:
#                 st.session_state.interpolated_eta = None

#         st.success("Done!")

# # === Display ===
# if st.session_state.interpolated_T is not None:
#     interpolated_T = st.session_state.interpolated_T
#     interpolated_eta = st.session_state.interpolated_eta
#     results = st.session_state.results
#     sources = st.session_state.sources
#     Nt, Ny, Nx = st.session_state.shape

#     # Weights
#     st.subheader("Hybrid Attention Weights")
#     df_weights = pd.DataFrame({
#         'Source': [s['file_name'] for s in sources],
#         'P (W)': [s['P'] for s in sources],
#         'v (m/s)': [f"{s['v']:.3f}" for s in sources],
#         'Hybrid': np.round(results['combined_weights'], 4)
#     })
#     st.dataframe(df_weights.style.bar(subset=['Hybrid'], color='#5fba7d'))

#     # Tabs
#     tab_list = ["Temperature"]
#     if interpolated_eta is not None:
#         tab_list.append("Phase Field η")

#     tabs = st.tabs(tab_list)

#     with tabs[0]:
#         st.subheader("Temperature Field")
#         t_min, t_max = interpolated_T.min(), interpolated_T.max()

#         frames = []
#         for k in range(Nt):
#             display_slice = np.flipud(interpolated_T[k])
#             frames.append(go.Frame(data=go.Heatmap(z=display_slice, colorscale='Inferno', zmin=t_min, zmax=t_max)))

#         fig = go.Figure(data=frames[0].data, frames=frames)
#         fig.update_layout(title="Temperature Evolution", xaxis_title="X", yaxis_title="Y (top=surface)")
#         fig.update_yaxes(scaleanchor="x", scaleratio=1)
#         st.plotly_chart(fig)

#         with st.expander("Static Temperature"):
#             t_idx = st.slider("Time", 0, Nt-1, Nt//2, key="t_static")
#             slice_disp = np.flipud(interpolated_T[t_idx])
#             fig_static, ax = plt.subplots()
#             cont = ax.contourf(slice_disp, cmap='jet')
#             ax.set_aspect('equal')
#             plt.colorbar(cont)
#             st.pyplot(fig_static)

#     if interpolated_eta is not None:
#         with tabs[1]:
#             st.subheader("Phase Field η")
#             eta_min, eta_max = interpolated_eta.min(), interpolated_eta.max()

#             frames = []
#             for k in range(Nt):
#                 display_slice = np.flipud(interpolated_eta[k])
#                 frames.append(go.Frame(data=go.Heatmap(z=display_slice, colorscale='viridis', zmin=eta_min, zmax=eta_max)))

#             fig = go.Figure(data=frames[0].data, frames=frames)
#             fig.update_layout(title="Phase Field Evolution", xaxis_title="X", yaxis_title="Y (top=surface)")
#             fig.update_yaxes(scaleanchor="x", scaleratio=1)
#             st.plotly_chart(fig)

#             with st.expander("Static Phase Field"):
#                 t_idx = st.slider("Time", 0, Nt-1, Nt//2, key="eta_static")
#                 slice_disp = np.flipud(interpolated_eta[t_idx])
#                 fig_static, ax = plt.subplots()
#                 cont = ax.contourf(slice_disp, cmap='viridis')
#                 ax.set_aspect('equal')
#                 plt.colorbar(cont)
#                 st.pyplot(fig_static)

#     # Download
#     st.subheader("Download")
#     save_dict = {'T': interpolated_T, 'P': p_target, 'v': v_target}
#     if interpolated_eta is not None:
#         save_dict['eta'] = interpolated_eta
#     buffer = io.BytesIO()
#     np.savez_compressed(buffer, **save_dict)
#     buffer.seek(0)
#     st.download_button("Download .npz", buffer, f"interp_P{p_target}_v{v_target:.3f}.npz")






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

# === Initialize Session State ===
if 'interpolated_T' not in st.session_state:
    st.session_state.interpolated_T = None
if 'interpolated_eta' not in st.session_state:
    st.session_state.interpolated_eta = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'shape' not in st.session_state:
    st.session_state.shape = None
if 'sources' not in st.session_state:
    st.session_state.sources = None
if 'computed_params' not in st.session_state:
    st.session_state.computed_params = None

# === Page Config ===
st.set_page_config(page_title="Attention-Based Multi-Field Interpolation", layout="wide")
st.title("Attention-Driven Interpolation of Temperature & Phase Field")
st.markdown("""
**Scientific Context**: Interpolates temperature \( T(t, y, x) \) and phase field \( \eta(t, y, x) \) using hybrid attention weights from (P, v).
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
    p_target = st.number_input("Target Laser Power \( P^* \) (W)", 100.0, 1000.0, 350.0, 10.0)
    v_target = st.number_input("Target Scan Velocity \( v^* \) (m/s)", 0.1, 3.0, 0.5, 0.05)

    st.header("Post-Processing")
    clip_eta = st.checkbox("Clip η to [0,1] (rectify max<1 artifact)", value=True)

# === Folders ===
t_folder = st.text_input("Temperature folder path", "t_folder")
eta_folder = st.text_input("Phase field folder path", "eta_folder")

sources = []
loaded_T_arrays = []
loaded_eta_arrays = []
param_list = []

if os.path.isdir(t_folder):
    npy_paths = sorted([os.path.join(t_folder, f) for f in os.listdir(t_folder) if f.lower().endswith('.npy')])
    st.info(f"Found {len(npy_paths)} temperature files")

    for path in npy_paths:
        filename = os.path.basename(path).lower()
        match = re.search(r'tsol(\d+)w(\d+)v\.npy$', filename)
        if not match:
            st.warning(f"Skipping {filename}")
            continue

        p_str, v_digits = match.groups()
        p = int(p_str)
        v = int(v_digits) / (10 ** (len(v_digits) - 1))

        try:
            T_arr = np.load(path)
            if T_arr.ndim != 3:
                st.error(f"{filename}: Expected 3D")
                continue

            eta_arr = None
            if os.path.isdir(eta_folder):
                eta_path = os.path.join(eta_folder, f"ETAsol{p_str}W{v_digits}V.npy")
                if os.path.exists(eta_path):
                    eta_arr = np.load(eta_path)
                    if eta_arr.shape != T_arr.shape:
                        st.warning(f"ETA shape mismatch for {filename}")
                        eta_arr = None

            sources.append({'file_name': os.path.basename(path), 'P': p, 'v': v, 'shape': T_arr.shape, 'eta': eta_arr is not None})
            loaded_T_arrays.append(T_arr)
            loaded_eta_arrays.append(eta_arr)
            param_list.append((p, v))

        except Exception as e:
            st.error(f"Error {filename}: {e}")
else:
    st.error(f"Temperature folder '{t_folder}' not found")

if sources:
    df_sources = pd.DataFrame(sources)[['file_name', 'P', 'v', 'eta', 'shape']]
    st.dataframe(df_sources.style.format({'v': '{:.3f}'}))

    if len(set(s['shape'] for s in sources)) != 1:
        st.error("Shapes must match")
        loaded_T_arrays = []

# === Interpolation ===
current_params = (p_target, v_target, sigma, num_heads, d_head, clip_eta, tuple(param_list))
if st.button("Run Interpolation", type="primary") or (st.session_state.interpolated_T is not None and st.session_state.computed_params != current_params):
    if len(loaded_T_arrays) < 2:
        st.error("Need ≥2 sources")
    else:
        with st.spinner("Interpolating..."):
            shape = loaded_T_arrays[0].shape
            Nt, Ny, Nx = shape

            interpolator = MultiParamAttentionInterpolator(sigma=sigma, num_heads=num_heads, d_head=d_head)
            results = interpolator.compute_weights(param_list, p_target, v_target)
            weights = results['combined_weights']

            interpolated_T = np.sum([w * arr for w, arr in zip(weights, loaded_T_arrays)], axis=0)

            st.session_state.interpolated_T = interpolated_T
            st.session_state.results = results
            st.session_state.shape = shape
            st.session_state.sources = sources
            st.session_state.computed_params = current_params

            if all(e is not None for e in loaded_eta_arrays):
                interpolated_eta = np.sum([w * arr for w, arr in zip(weights, loaded_eta_arrays)], axis=0)
                if clip_eta:
                    interpolated_eta = np.clip(interpolated_eta, 0.0, 1.0)
                    st.info("Applied clipping: η forced to [0,1]")
                st.session_state.interpolated_eta = interpolated_eta
            else:
                st.session_state.interpolated_eta = None

        st.success("Done!")

# === Display ===
if st.session_state.interpolated_T is not None:
    interpolated_T = st.session_state.interpolated_T
    interpolated_eta = st.session_state.interpolated_eta
    results = st.session_state.results
    sources = st.session_state.sources
    Nt, Ny, Nx = st.session_state.shape

    st.subheader("Hybrid Attention Weights")
    df_weights = pd.DataFrame({
        'Source': [s['file_name'] for s in sources],
        'P (W)': [s['P'] for s in sources],
        'v (m/s)': [f"{s['v']:.3f}" for s in sources],
        'Hybrid': np.round(results['combined_weights'], 4)
    })
    st.dataframe(df_weights.style.bar(subset=['Hybrid'], color='#5fba7d'))

    tab_list = ["Temperature"]
    if interpolated_eta is not None:
        tab_list.append("Phase Field η")

    tabs = st.tabs(tab_list)

    with tabs[0]:
        st.subheader("Temperature Field")
        t_min, t_max = interpolated_T.min(), interpolated_T.max()

        frames = []
        for k in range(Nt):
            display_slice = np.flipud(interpolated_T[k])
            frames.append(go.Frame(data=go.Heatmap(z=display_slice, colorscale='Inferno', zmin=t_min, zmax=t_max)))

        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(title="Temperature Evolution", xaxis_title="X", yaxis_title="Y (top=surface)")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig)

        with st.expander("Static Temperature"):
            t_idx = st.slider("Time", 0, Nt-1, Nt//2, key="t_static")
            slice_disp = np.flipud(interpolated_T[t_idx])
            fig_static, ax = plt.subplots()
            cont = ax.contourf(slice_disp, cmap='jet',)
            ax.set_aspect('equal')
            plt.colorbar(cont)
            st.pyplot(fig_static)

    if interpolated_eta is not None:
        with tabs[1]:
            st.subheader("Phase Field η")
            eta_min, eta_max = interpolated_eta.min(), interpolated_eta.max()
            st.info(f"η range after processing: [{eta_min:.3f}, {eta_max:.3f}]")

            frames = []
            for k in range(Nt):
                display_slice = np.flipud(interpolated_eta[k])
                frames.append(go.Frame(data=go.Heatmap(z=display_slice, colorscale='viridis', zmin=0.0, zmax=1.0)))

            fig = go.Figure(data=frames[0].data, frames=frames)
            fig.update_layout(title="Phase Field Evolution", xaxis_title="X", yaxis_title="Y (top=surface)")
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig)

            with st.expander("Static Phase Field"):
                t_idx = st.slider("Time", 0, Nt-1, Nt//2, key="eta_static")
                slice_disp = np.flipud(interpolated_eta[t_idx])
                fig_static, ax = plt.subplots()
                cont = ax.contourf(slice_disp, levels=20, cmap='viridis',)
                ax.set_aspect('equal')
                plt.colorbar(cont, label="η (clipped [0,1])")
                st.pyplot(fig_static)

    # Download
    st.subheader("Download")
    save_dict = {'T': interpolated_T, 'P': p_target, 'v': v_target}
    if interpolated_eta is not None:
        save_dict['eta'] = interpolated_eta
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **save_dict)
    buffer.seek(0)
    st.download_button("Download .npz", buffer, f"interp_P{p_target}_v{v_target:.3f}.npz")

else:
    st.info("Load sources and run interpolation.")