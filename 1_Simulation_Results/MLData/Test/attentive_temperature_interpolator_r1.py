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
from typing import Dict, List, Optional, Tuple
# ============================================================================
# SCRIPT DIRECTORY RESOLUTION (for reliable relative path loading)
# ============================================================================
# Resolve folder paths relative to the script location, not the CWD.
# This ensures folders like TEMP/, COMP/, ETALIQ/ are found regardless
# of where streamlit is launched from.
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(SCRIPT_DIR, "TEMP")
COMP_DIR = os.path.join(SCRIPT_DIR, "COMP")
ETALIQ_DIR = os.path.join(SCRIPT_DIR, "ETALIQ")


# ============================================================================
# GLOBAL CONFIGURATION: cTF LOOKUP TABLE FOR INITIAL COMPOSITIONS
# ============================================================================
# This table defines the INITIAL mole fractions [x_Co, x_Cr, x_Fe, x_Ni] 
# assigned to LIQUID (η=1) and FCC (η=0) end-members for each TF index.
# These are NOT solution fields - they are initial condition parameters
# used to construct the spatial cTau tensor via: 
# c_init(y,x) = η(0,y,x) * c_LIQUID + (1-η(0,y,x)) * c_FCC
# ============================================================================
CTF_TABLE: Dict[int, Dict[str, List[float]]] = {
    0: {
        'liquid': [0.35, 0.13, 0.15, 0.37],  # ETA=1: Co, Cr, Fe, Ni mole fractions
        'fcc':    [0.28, 0.18, 0.22, 0.32]   # ETA=0: Co, Cr, Fe, Ni mole fractions
    },
    1: {
        'liquid': [0.32, 0.15, 0.17, 0.36],
        'fcc':    [0.25, 0.21, 0.25, 0.29]
    },
    2: {
        'liquid': [0.30, 0.19, 0.11, 0.40],
        'fcc':    [0.25, 0.23, 0.20, 0.32]
    },
    3: {
        'liquid': [0.33, 0.11, 0.23, 0.33],
        'fcc':    [0.31, 0.15, 0.27, 0.27]
    },
    4: {
        'liquid': [0.38, 0.17, 0.19, 0.26],
        'fcc':    [0.30, 0.22, 0.26, 0.22]
    }
}

ELEMENTS = ['Co', 'Cr', 'Fe', 'Ni']

# ============================================================================
# TIME ARRAYS FOR DIFFERENT SCAN SPEEDS (for visualization annotations)
# ============================================================================
TIME45 = np.arange(30.5, 1648.5+2, 2)[0::81]
TIME50 = np.arange(30.5, 1518.5+2, 2)[0::75]
TIME60 = np.arange(30.5, 1268.5+2, 2)[0::62]
TIME70 = np.arange(30.5, 1088.5+2, 2)[0::53]

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'interpolated_T' not in st.session_state:
    st.session_state.interpolated_T = None
if 'interpolated_COMP' not in st.session_state:
    st.session_state.interpolated_COMP = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'shape' not in st.session_state:
    st.session_state.shape = None
if 'sources' not in st.session_state:
    st.session_state.sources = None
if 'computed_params' not in st.session_state:
    st.session_state.computed_params = None
if 'field_type' not in st.session_state:
    st.session_state.field_type = 'temperature'  # 'temperature' or 'composition'

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Phase-Field Aware Attention Interpolation", layout="wide")
st.title("🔬 Attention-Driven Interpolation for Laser Processed HEAs")
st.markdown("""
### Scientific Context
This tool implements **transformer-inspired cross-attention with Gaussian spatial locality regularization** 
to interpolate 3D spatiotemporal fields from precomputed phase-field/FEM simulations of Co-Cr-Fe-Ni 
high-entropy alloys under laser processing.

#### 🔑 Key Architecture Concepts:
1. **Filename Convention**: `p{i}s{j}cTF{k}.npy`
   - `p{i}`: Laser power in Watts (e.g., p350 → 350 W)
   - `s{j}`: Scan speed in cm/s (e.g., s45 → 45 cm/s)  
   - `cTF{k}`: **Initial composition tensor index** (k=0,1,2,3,4) encoding mole fractions for LIQUID/FCC end-members

2. **cTau Tensor Construction** (Initial Conditions ONLY):
   ```
   c_init(y,x) = η(0,y,x) · c_LIQUID_TFk + (1-η(0,y,x)) · c_FCC_TFk
   ```
   - `η(0,y,x)`: Phase field from ETALIQ container at t=0 (1=LIQUID, 0=FCC, 0<η<1=interface)
   - `c_LIQUID_TFk`, `c_FCC_TFk`: Mole fraction vectors from cTF lookup table
   - ⚠️ These are INITIAL PARAMETERS, not evolved solution fields

3. **Solution Field Folders** (Interpolated Independently):
   - `TEMP/`: Temperature fields T(t,y,x) [K]
   - `COMP/`: Composition fields x_Co(t,y,x), x_Cr(t,y,x), ... [mole fraction]
   - `ETALIQ/`: Phase field η(t,y,x) ∈ [0,1] (foundational for initial condition construction)

4. **Interpolation Strategy**:
   - Temperature: Interpolate using (P, v) proximity (compositions weakly coupled)
   - Compositions: Interpolate using (P, v, TF_idx) joint proximity (initial conditions matter)
   - ETALIQ: Loaded directly or interpolated separately if surrogate phase field needed
""")

# ============================================================================
# ATTENTION MODEL DEFINITION (UPGRADED FOR 3D PARAMETER SPACE)
# ============================================================================
class PhaseAwareAttentionInterpolator(nn.Module):
    """
    Multi-parameter attention interpolator with optional categorical TF index embedding.
    
    Parameters:
    - sigma_param: Gaussian width for (P,v) parameter space locality
    - sigma_comp: Gaussian width for composition similarity (used when TF-aware)
    - num_heads: Number of attention heads
    - d_head: Dimension per attention head
    - d_emb_tf: Embedding dimension for categorical TF index (0-4)
    - use_tf_embedding: Whether to include TF index in attention computation
    """
    def __init__(self, sigma_param=0.2, sigma_comp=0.05, 
                 num_heads=4, d_head=8, d_emb_tf=4, use_tf_embedding=True):
        super().__init__()
        self.sigma_param = sigma_param
        self.sigma_comp = sigma_comp
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_emb_tf = d_emb_tf
        self.use_tf_embedding = use_tf_embedding
        
        # Projection layers for continuous process parameters (P, v)
        self.W_q_proc = nn.Linear(2, num_heads * d_head, bias=False)
        self.W_k_proc = nn.Linear(2, num_heads * d_head, bias=False)
        
        # Embedding and projection for categorical TF index (if enabled)
        if use_tf_embedding:
            self.emb_tf = nn.Embedding(5, d_emb_tf)  # 5 TF indices: 0,1,2,3,4
            self.W_q_comp = nn.Linear(d_emb_tf + 4, d_emb_tf, bias=False)  # +4 for composition vector similarity
            self.W_k_comp = nn.Linear(d_emb_tf + 4, d_emb_tf, bias=False)
    
    def normalize_params(self, params_list: List[Tuple[float, float]], 
                        target_params: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Min-max normalize process parameters (P, v) to [0,1]"""
        all_params = np.array(params_list + [target_params])
        mins = all_params.min(axis=0)
        maxs = all_params.max(axis=0)
        range_ = maxs - mins + 1e-8
        norm_sources = (np.array(params_list) - mins) / range_
        norm_target = (np.array(target_params) - mins) / range_
        return torch.tensor(norm_sources, dtype=torch.float32), torch.tensor(norm_target, dtype=torch.float32)
    
    def compute_composition_similarity(self, src_compositions: Dict[str, List[float]], 
                                      tgt_compositions: Dict[str, List[float]]) -> float:
        """
        Compute Gaussian similarity between source and target composition sets.
        Averages similarity over LIQUID and FCC end-members.
        """
        def gaussian_sim(c1, c2, sigma):
            return np.exp(-np.sum((np.array(c1) - np.array(c2))**2) / (2 * sigma**2))
        
        sim_liquid = gaussian_sim(src_compositions['liquid'], tgt_compositions['liquid'], self.sigma_comp)
        sim_fcc = gaussian_sim(src_compositions['fcc'], tgt_compositions['fcc'], self.sigma_comp)
        return 0.5 * (sim_liquid + sim_fcc)
    
    def compute_weights(self, sources: List[Dict], target_params: Dict, 
                       field_type: str = 'temperature') -> Dict[str, np.ndarray]:
        """
        Compute hybrid attention weights for interpolation.
        
        Args:
            sources: List of source simulation metadata dicts
            target_params: Target parameter dict with 'P', 'v', 'TF_idx', 'compositions'
            field_type: 'temperature' (use P,v only) or 'composition' (use P,v,TF_idx)
        
        Returns:
            Dict with attention_weights, spatial_weights, combined_weights, etc.
        """
        N = len(sources)
        if N == 0:
            return {}
        
        # Extract process parameters
        src_proc_params = [(s['P'], s['v']) for s in sources]
        tgt_proc_params = (target_params['P'], target_params['v'])
        
        # Normalize process parameters
        src_proc_tensor, tgt_proc_tensor_1d = self.normalize_params(src_proc_params, tgt_proc_params)
        tgt_proc_tensor = tgt_proc_tensor_1d.unsqueeze(0)  # (1, 2)
        
        # === STEP 1: Process Parameter Attention (P, v) ===
        q_proc = self.W_q_proc(tgt_proc_tensor).view(1, self.num_heads, self.d_head)
        k_proc = self.W_k_proc(src_proc_tensor).view(N, self.num_heads, self.d_head)
        
        attn_logits = torch.einsum('nhd,mhd->nmh', k_proc, q_proc) / np.sqrt(self.d_head)
        attn_weights_proc = torch.softmax(attn_logits.squeeze(1), dim=0)  # (N,)
        attn_weights_proc = attn_weights_proc.mean(dim=1)  # Average over heads
        
        # === STEP 2: Gaussian Spatial Locality in (P,v) Space ===
        diffs = src_proc_tensor - tgt_proc_tensor
        dists = torch.sqrt(torch.sum(diffs**2, dim=1))
        spatial_weights_proc = torch.exp(-dists**2 / (2 * self.sigma_param**2))
        spatial_weights_proc = spatial_weights_proc / (spatial_weights_proc.sum() + 1e-8)
        
        # === STEP 3: Composition-Aware Weighting (if field_type == 'composition') ===
        if field_type == 'composition' and self.use_tf_embedding:
            comp_similarities = []
            for src in sources:
                sim = self.compute_composition_similarity(
                    src['compositions'], target_params['compositions']
                )
                comp_similarities.append(sim)
            comp_weights = torch.tensor(comp_similarities, dtype=torch.float32)
            comp_weights = comp_weights / (comp_weights.sum() + 1e-8)
        else:
            # For temperature or if TF-embedding disabled: uniform composition weight
            comp_weights = torch.ones(N, dtype=torch.float32) / N
        
        # === STEP 4: Hybrid Weight Combination ===
        combined = attn_weights_proc * spatial_weights_proc * comp_weights
        combined = combined / (combined.sum() + 1e-8)
        
        return {
            'attention_weights_proc': attn_weights_proc.detach().numpy(),
            'spatial_weights_proc': spatial_weights_proc.detach().numpy(),
            'composition_weights': comp_weights.detach().numpy(),
            'combined_weights': combined.detach().numpy(),
            'norm_sources_proc': src_proc_tensor.numpy(),
            'norm_target_proc': tgt_proc_tensor_1d.numpy(),
            'W_q_proc': self.W_q_proc.weight.data.numpy(),
            'W_k_proc': self.W_k_proc.weight.data.numpy()
        }

# ============================================================================
# FILENAME PARSING UTILITIES
# ============================================================================
def parse_tensor_filename(filename: str) -> Optional[Dict]:
    """
    Parse filename pattern: p{i}s{j}cTF{k}.npy
    
    Returns dict with:
    - P: laser power in Watts
    - v: scan speed in m/s (converted from cm/s)
    - TF_idx: composition tensor index (0-4)
    - compositions: dict with 'liquid' and 'fcc' mole fraction lists from CTF_TABLE
    """
    # Use case-insensitive regex on original filename to match p350s45cTF0.npy patterns
    match = re.search(r'p(\d+)s(\d+)cTF(\d+)\.npy$', filename, re.IGNORECASE)
    
    if not match:
        return None
    
    power = int(match.group(1))           # e.g., 350 W
    speed_cm_s = int(match.group(2))       # e.g., 45 cm/s
    tf_idx = int(match.group(3))           # e.g., 0, 1, 2, 3, or 4
    
    # Validate TF index
    if tf_idx not in CTF_TABLE:
        return None
    
    # Convert speed: cm/s → m/s
    speed_m_s = speed_cm_s / 100.0
    
    return {
        'P': power,
        'v': speed_m_s,
        'speed_cm_s': speed_cm_s,
        'TF_idx': tf_idx,
        'compositions': CTF_TABLE[tf_idx].copy()  # Return copy to avoid mutation
    }

def format_composition_vector(comp_list: List[float], prefix: str = "") -> str:
    """Format mole fraction list as readable string"""
    return ", ".join([f"{prefix}{ELEMENTS[i]}={val:.2f}" for i, val in enumerate(comp_list)])

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_simulation_metadata(folder_path: str, field_type: str) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Load simulation files from specified folder with metadata parsing.
    
    Args:
        folder_path: Path to folder containing .npy files (TEMP/, COMP/, etc.)
        field_type: 'temperature' or 'composition' (affects weight computation later)
    
    Returns:
        sources: List of metadata dicts for each valid file
        loaded_arrays: List of numpy arrays (solution fields)
    """
    sources = []
    loaded_arrays = []
    
    if not os.path.isdir(folder_path):
        return sources, loaded_arrays
    
    npy_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.npy')])
    
    for filename in npy_files:
        parsed = parse_tensor_filename(filename)
        if parsed is None:
            st.warning(f"⚠️ Skipping '{filename}': does not match pattern p<power>s<speed>cTF<idx>.npy (case-insensitive)")
            continue
        
        try:
            file_path = os.path.join(folder_path, filename)
            field_array = np.load(file_path)
            
            # Validate array dimensions: should be 3D (Nt, Ny, Nx)
            if field_array.ndim != 3:
                st.error(f"❌ '{filename}': Expected 3D array (Nt, Ny, Nx), got shape {field_array.shape}")
                continue
            
            # Store metadata with cTau tensor elaboration
            source_meta = {
                'file_name': filename,
                'P': parsed['P'],
                'v': parsed['v'],
                'speed_cm_s': parsed['speed_cm_s'],
                'TF_idx': parsed['TF_idx'],
                'compositions': parsed['compositions'],
                'shape': field_array.shape,
                # Explicit cTau tensor description for UI display
                'cTau_description': {
                    'meaning': 'Initial condition tensor: spatial mole fractions at t=0',
                    'construction': 'c_init(y,x) = η(0,y,x)·c_LIQUID + (1-η(0,y,x))·c_FCC',
                    'liquid_mole_fractions': dict(zip(ELEMENTS, parsed['compositions']['liquid'])),
                    'fcc_mole_fractions': dict(zip(ELEMENTS, parsed['compositions']['fcc'])),
                    'note': 'These are INITIAL PARAMETERS only. Evolved composition fields are in COMP/ folder.'
                }
            }
            
            sources.append(source_meta)
            loaded_arrays.append(field_array)
            
            st.success(f"✅ Loaded '{filename}' → P={parsed['P']}W, v={parsed['speed_cm_s']}cm/s, TF={parsed['TF_idx']}")
            
        except Exception as e:
            st.error(f"❌ Error loading '{filename}': {str(e)}")
            continue
    
    return sources, loaded_arrays

def load_etaliq_phase_field(etaliq_folder: str, filename: str) -> Optional[np.ndarray]:
    """
    Load corresponding ETALIQ phase field for a given simulation filename.
    Returns η(t,y,x) array or None if not found.
    """
    etaliq_path = os.path.join(etaliq_folder, filename)
    if os.path.exists(etaliq_path):
        try:
            eta_array = np.load(etaliq_path)
            return eta_array
        except Exception as e:
            st.warning(f"⚠️ Could not load ETALIQ file '{filename}': {e}")
    return None

# ============================================================================
# SIDEBAR: MODEL CONTROLS & TARGET PARAMETERS
# ============================================================================
with st.sidebar:
    st.header("⚙️ Attention Model Configuration")

    # Show resolved folder paths
    with st.expander("📁 Resolved Folder Paths", expanded=False):
        st.code(f"SCRIPT_DIR = {SCRIPT_DIR}")
        st.code(f"TEMP_DIR = {TEMP_DIR}")
        st.code(f"COMP_DIR = {COMP_DIR}")
        st.code(f"ETALIQ_DIR = {ETALIQ_DIR}")
        st.caption("Folders are resolved relative to the script file location.")


    
    sigma_param = st.slider("Gaussian Locality σ (P,v space)", 0.05, 0.50, 0.20, 0.01, 
                           help="Width of Gaussian prior favoring sources close in (Power, Velocity) space")
    sigma_comp = st.slider("Composition Similarity σ", 0.01, 0.20, 0.05, 0.01,
                          help="Width of Gaussian for composition mole fraction similarity (used for composition fields)")
    num_heads = st.slider("Attention Heads", 1, 8, 4)
    d_head = st.slider("Dimension per Head", 4, 16, 8)
    d_emb_tf = st.slider("TF Index Embedding Dim", 2, 8, 4)
    use_tf_embedding = st.checkbox("Use TF Index in Attention", value=True,
                                  help="Include composition tensor index in weight computation (essential for composition field interpolation)")
    
    seed = st.number_input("Random Seed", 0, 9999, 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    st.header("🎯 Target Simulation Parameters")
    
    # Field type selector
    field_type = st.radio("Field to Interpolate", 
                         options=['temperature', 'composition'],
                         format_func=lambda x: "🌡️ Temperature T(t,y,x)" if x=='temperature' else "🧪 Composition x_m(t,y,x)",
                         help="Temperature fields depend primarily on (P,v); Composition fields depend on (P,v) + initial composition (TF index)")
    st.session_state.field_type = field_type
    
    # Parameter inputs with context-aware ranges
    if field_type == 'temperature':
        st.info("🌡️ Temperature interpolation uses (P, v) proximity. Initial compositions (cTF) have weak coupling.")
    else:
        st.info("🧪 Composition interpolation uses (P, v, TF_idx) joint proximity. Initial mole fractions strongly affect diffusion.")
    
    p_target = st.number_input("Target Laser Power P* (W)", 300.0, 800.0, 420.0, 10.0)
    v_target_cm_s = st.number_input("Target Scan Speed v* (cm/s)", 40.0, 80.0, 45.0, 5.0)
    v_target_m_s = v_target_cm_s / 100.0  # Convert to m/s for internal use
    
    # TF index selector (only relevant for composition fields)
    if field_type == 'composition':
        tf_target = st.selectbox("Target Initial Composition Tensor (cTF index)", 
                                options=[0,1,2,3,4],
                                format_func=lambda k: f"TF{k}: {format_composition_vector(CTF_TABLE[k]['liquid'], 'L:')} | {format_composition_vector(CTF_TABLE[k]['fcc'], 'S:')}")
    else:
        tf_target = 0  # Dummy value for temperature mode
    
    target_params = {
        'P': p_target,
        'v': v_target_m_s,
        'speed_cm_s': v_target_cm_s,
        'TF_idx': tf_target,
        'compositions': CTF_TABLE[tf_target]
    }

# ============================================================================
# MAIN: DATA LOADING & SOURCE SELECTION
# ============================================================================
st.subheader("📁 Load Simulation Data")

# Folder configuration with clear labeling
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    temp_folder = st.text_input("🌡️ Temperature Folder (TEMP/)", TEMP_DIR, help=f"Default: {TEMP_DIR}")
with col_f2:
    comp_folder = st.text_input("🧪 Composition Folder (COMP/)", COMP_DIR, help=f"Default: {COMP_DIR}")
with col_f3:
    etaliq_folder = st.text_input("🌀 ETALIQ Phase Field Folder", ETALIQ_DIR, help=f"Default: {ETALIQ_DIR}")

load_method = st.selectbox("Loading Method", ["From local folders", "Upload files manually"])

sources = []
loaded_arrays = []
param_list = []

if load_method == "From local folders":
    # Determine which folder to load based on field type
    if field_type == 'temperature':
        target_folder = temp_folder
        folder_label = "Temperature (TEMP/)"
    else:
        target_folder = comp_folder
        folder_label = "Composition (COMP/)"

    # Auto-create folder if it doesn't exist (relative to script location)
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder, exist_ok=True)
        st.info(f"📁 Created folder: `{target_folder}` — please place .npy files there.")

    npy_count = len([f for f in os.listdir(target_folder) if f.lower().endswith('.npy')])
    st.info(f"📂 Found {npy_count} .npy files in {folder_label}: `{target_folder}`")

    if npy_count > 0:
        with st.spinner(f"Loading {folder_label} files..."):
            sources, loaded_arrays = load_simulation_metadata(target_folder, field_type)
    else:
        st.warning(f"⚠️ No .npy files found in '{target_folder}'. Please add simulation data.")

else:  # Upload files manually
    uploaded_files = st.file_uploader(
        f"Upload .npy files containing 3D {'temperature' if field_type=='temperature' else 'composition'} fields T(t,y,x) or x_m(t,y,x)",
        type=["npy"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            parsed = parse_tensor_filename(uploaded_file.name)
            if parsed is None:
                st.warning(f"⚠️ Skipping '{uploaded_file.name}': filename must match p*i*s*j*cTF*k.npy")
                continue
            
            try:
                uploaded_file.seek(0)
                field_array = np.load(uploaded_file)
                
                if field_array.ndim != 3:
                    st.error(f"❌ '{uploaded_file.name}': Expected 3D array (Nt, Ny, Nx), got {field_array.shape}")
                    continue
                
                source_meta = {
                    'file_name': uploaded_file.name,
                    'P': parsed['P'],
                    'v': parsed['v'],
                    'speed_cm_s': parsed['speed_cm_s'],
                    'TF_idx': parsed['TF_idx'],
                    'compositions': parsed['compositions'],
                    'shape': field_array.shape,
                    'cTau_description': {
                        'meaning': 'Initial condition tensor: spatial mole fractions at t=0',
                        'construction': 'c_init(y,x) = η(0,y,x)·c_LIQUID + (1-η(0,y,x))·c_FCC',
                        'liquid_mole_fractions': dict(zip(ELEMENTS, parsed['compositions']['liquid'])),
                        'fcc_mole_fractions': dict(zip(ELEMENTS, parsed['compositions']['fcc'])),
                        'note': 'These are INITIAL PARAMETERS only. Evolved composition fields are in COMP/ folder.'
                    }
                }
                
                sources.append(source_meta)
                loaded_arrays.append(field_array)
                st.success(f"✅ Loaded '{uploaded_file.name}'")
                
            except Exception as e:
                st.error(f"❌ Error loading '{uploaded_file.name}': {e}")

# ============================================================================
# DISPLAY LOADED SOURCES WITH cTAU TENSOR ELABORATION
# ============================================================================
if sources:
    st.subheader(f"📋 Loaded {len(sources)} Valid Source Simulations")
    
    # Create dataframe with key metadata
    df_sources = pd.DataFrame([{
        'File': s['file_name'],
        'Power (W)': s['P'],
        'Speed (cm/s)': s['speed_cm_s'],
        'TF Index': s['TF_idx'],
        'Shape': '×'.join(map(str, s['shape'])),
        'Initial Co (L/S)': f"{s['compositions']['liquid'][0]:.2f}/{s['compositions']['fcc'][0]:.2f}",
        'Initial Cr (L/S)': f"{s['compositions']['liquid'][1]:.2f}/{s['compositions']['fcc'][1]:.2f}",
        'Initial Fe (L/S)': f"{s['compositions']['liquid'][2]:.2f}/{s['compositions']['fcc'][2]:.2f}",
        'Initial Ni (L/S)': f"{s['compositions']['liquid'][3]:.2f}/{s['compositions']['fcc'][3]:.2f}",
    } for s in sources])
    
    st.dataframe(df_sources, use_container_width=True)
    
    # Shape consistency check
    shapes = [s['shape'] for s in sources]
    if len(set(shapes)) != 1:
        st.error(f"❌ Shape mismatch: Sources have different shapes {set(shapes)}. All arrays must be (Nt, Ny, Nx).")
        loaded_arrays = []
        sources = []
    else:
        st.success(f"✅ All sources consistent: {shapes[0]} = (Nt time steps, Ny y-points, Nx x-points)")
    
    # 🔍 EXPANDABLE: Detailed cTau Tensor Inspection for Each Source
    with st.expander("🔬 Expand: Detailed cTau Tensor Inspection for Loaded Sources", expanded=False):
        st.markdown("""
        #### Understanding the cTau Tensor in Filenames
        The `cTFk` suffix in filenames (e.g., `p350s45cTF2.npy`) encodes **initial condition parameters only**:
        
        - **NOT** a solution field that evolves in time
        - **NOT** the instantaneous composition at t>0
        - **IS** the mole fraction vectors assigned to LIQUID (η=1) and FCC (η=0) phases at t=0
        
        #### Spatial Construction Formula:
        ```
        c_init(y,x) = η(0,y,x) · c_LIQUID_TFk + (1 - η(0,y,x)) · c_FCC_TFk
        ```
        where η(0,y,x) is loaded from the ETALIQ container.
        """)
        
        for src in sources:
            with st.container():
                st.markdown(f"##### 📄 File: `{src['file_name']}`")
                col_c1, col_c2 = st.columns([1, 2])
                
                with col_c1:
                    st.markdown("**Process Parameters**")
                    st.write(f"- Power: {src['P']} W")
                    st.write(f"- Speed: {src['speed_cm_s']} cm/s = {src['v']:.3f} m/s")
                    st.write(f"- TF Index: {src['TF_idx']}")
                
                with col_c2:
                    st.markdown("**cTau Initial Composition Tensor**")
                    st.markdown(f"*Construction*: `c_init = η·c_LIQUID + (1-η)·c_FCC`")
                    
                    # Display mole fractions in table
                    comp_data = []
                    for elem, liquid_val, fcc_val in zip(ELEMENTS, 
                                                        src['compositions']['liquid'],
                                                        src['compositions']['fcc']):
                        comp_data.append({
                            'Element': elem,
                            'LIQUID (η=1)': liquid_val,
                            'FCC (η=0)': fcc_val,
                            'Difference': abs(liquid_val - fcc_val)
                        })
                    comp_df = pd.DataFrame(comp_data)
                    st.dataframe(comp_df, use_container_width=True)  # values already formatted as floats
                    
                    st.caption("⚠️ Reminder: These are INITIAL mole fractions at t=0. Evolved composition fields are stored separately in COMP/ folder.")
                
                st.divider()

# ============================================================================
# INTERPOLATION EXECUTION
# ============================================================================
current_params = (p_target, v_target_m_s, tf_target, sigma_param, sigma_comp, 
                  num_heads, d_head, d_emb_tf, use_tf_embedding, field_type, tuple((s['P'], s['v'], s['TF_idx']) for s in sources))

if st.button("🚀 Run Phase-Aware Attention Interpolation", type="primary", 
             disabled=len(loaded_arrays) < 2):
    
    if len(loaded_arrays) < 2:
        st.error("❌ Need at least 2 valid source simulations to interpolate.")
    else:
        with st.spinner(f"Computing {'temperature' if field_type=='temperature' else 'composition'} interpolation with phase-aware attention..."):
            shape = loaded_arrays[0].shape
            Nt, Ny, Nx = shape
            
            # Initialize model
            interpolator = PhaseAwareAttentionInterpolator(
                sigma_param=sigma_param,
                sigma_comp=sigma_comp,
                num_heads=num_heads,
                d_head=d_head,
                d_emb_tf=d_emb_tf,
                use_tf_embedding=use_tf_embedding and field_type=='composition'
            )
            
            # Compute hybrid weights
            results = interpolator.compute_weights(sources, target_params, field_type=field_type)
            weights = results['combined_weights']  # Shape: (N_sources,)
            
            # Perform weighted interpolation
            interpolated_field = np.zeros(shape, dtype=np.float64)
            for w, arr in zip(weights, loaded_arrays):
                interpolated_field += w * arr
            
            # Cache results
            if field_type == 'temperature':
                st.session_state.interpolated_T = interpolated_field
            else:
                st.session_state.interpolated_COMP = interpolated_field
                
            st.session_state.results = results
            st.session_state.shape = shape
            st.session_state.sources = sources
            st.session_state.computed_params = current_params
            
            st.success(f"✅ Interpolation complete! Field shape: {shape}")

# ============================================================================
# RESULTS DISPLAY & VISUALIZATION
# ============================================================================
# Determine which interpolated field to display
if field_type == 'temperature' and st.session_state.interpolated_T is not None:
    interpolated_field = st.session_state.interpolated_T
    field_label = "Temperature T(t,y,x) [K]"
    field_unit = "K"
    cmap = 'Inferno'
elif field_type == 'composition' and st.session_state.interpolated_COMP is not None:
    interpolated_field = st.session_state.interpolated_COMP
    field_label = "Composition Field x_m(t,y,x) [mole fraction]"
    field_unit = "mol frac"
    cmap = 'viridis'
else:
    interpolated_field = None

if interpolated_field is not None:
    results = st.session_state.results
    shape = st.session_state.shape
    sources = st.session_state.sources
    Nt, Ny, Nx = shape
    
    st.subheader(f"📊 Results: Interpolated {field_label}")
    st.info(f"Target: P* = {p_target} W, v* = {v_target_cm_s} cm/s" + 
            (f", TF* = {tf_target}" if field_type=='composition' else ""))
    st.info(f"Field dimensions: {Nt} time steps × {Ny} y-points × {Nx} x-points")
    
    # === COLUMN 1: Weight Analysis ===
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.markdown("#### 🎯 Hybrid Attention Weights")
        
        # Weight breakdown table
        df_weights = pd.DataFrame({
            'Source File': [s['file_name'] for s in sources],
            'P (W)': [s['P'] for s in sources],
            'v (cm/s)': [f"{s['speed_cm_s']:.0f}" for s in sources],
            'TF': [s['TF_idx'] for s in sources],
            'Process Attention': np.round(results['attention_weights_proc'], 4),
            'Gaussian Locality': np.round(results['spatial_weights_proc'], 4),
        })
        
        if field_type == 'composition':
            df_weights['Composition Similarity'] = np.round(results['composition_weights'], 4)
        
        df_weights['Final Hybrid Weight'] = np.round(results['combined_weights'], 4)
        
        # Color-bar for final weights
        st.dataframe(
            df_weights.style.bar(subset=['Final Hybrid Weight'], color='#4CAF50', vmin=0, vmax=1),
            use_container_width=True
        )
        
        # Parameter space scatter plot
        fig_param = go.Figure()
        src_norm = results['norm_sources_proc']
        tgt_norm = results['norm_target_proc']
        
        # Source points colored by weight
        fig_param.add_trace(go.Scatter(
            x=src_norm[:,0], y=src_norm[:,1],
            mode='markers+text',
            marker=dict(
                size=20, 
                color=results['combined_weights'], 
                colorscale='Viridis', 
                showscale=True,
                colorbar=dict(title="Weight")
            ),
            text=[f"P={s['P']}W<br>v={s['speed_cm_s']}cm/s<br>TF={s['TF_idx']}" for s in sources],
            textposition="top center",
            name='Sources'
        ))
        
        # Target point
        fig_param.add_trace(go.Scatter(
            x=[tgt_norm[0]], y=[tgt_norm[1]],
            mode='markers+text',
            marker=dict(size=30, symbol='star', color='red', line=dict(width=2, color='darkred')),
            text=f"TARGET<br>P={p_target}W<br>v={v_target_cm_s}cm/s" + (f"<br>TF={tf_target}" if field_type=='composition' else ""),
            textposition="bottom center",
            name='Target'
        ))
        
        fig_param.update_layout(
            title="Normalized Parameter Space (Power vs Velocity)",
            xaxis_title="Normalized Laser Power",
            yaxis_title="Normalized Scan Speed",
            hovermode='closest'
        )
        st.plotly_chart(fig_param, use_container_width=True)
    
    # === COLUMN 2: Model Internals ===
    with col2:
        st.markdown("#### 🧠 Learned Projection Matrices")
        fig_proj, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        
        # Query weights heatmap
        sns.heatmap(results['W_q_proc'], ax=ax1, cmap='coolwarm', center=0, cbar=False)
        ax1.set_title("$W_q$ (Query Projection)")
        ax1.set_xlabel("Head×Dim")
        ax1.set_ylabel("Param (P,v)")
        
        # Key weights heatmap
        sns.heatmap(results['W_k_proc'], ax=ax2, cmap='coolwarm', center=0)
        ax2.set_title("$W_k$ (Key Projection)")
        ax2.set_xlabel("Head×Dim")
        ax2.set_ylabel("Param (P,v)")
        
        plt.tight_layout()
        st.pyplot(fig_proj)
        
        # Composition similarity note (if applicable)
        if field_type == 'composition' and use_tf_embedding:
            st.markdown("#### 🧪 Composition-Aware Weighting")
            st.write("Weights modulated by Gaussian similarity of initial mole fractions:")
            for i, src in enumerate(sources):
                sim = results['composition_weights'][i]
                st.write(f"- `{src['file_name']}` (TF{src['TF_idx']}): similarity = {sim:.3f}")
    
    # === FIELD VISUALIZATION ===
    st.divider()
    st.subheader(f"🎬 Animated {field_label} Evolution")
    
    # Time step selector
    t_idx = st.slider("Time Step Index", 0, Nt-1, Nt//2, key="anim_time")
    
    # Get current slice and statistics
    current_slice = interpolated_field[t_idx, :, :]
    t_min, t_max = interpolated_field.min(), interpolated_field.max()
    slice_min, slice_max = current_slice.min(), current_slice.max()
    
    st.write(f"**Statistics @ t={t_idx}**: Min = {slice_min:.2f} {field_unit}, Max = {slice_max:.2f} {field_unit}, Mean = {current_slice.mean():.2f} {field_unit}")
    
    # Animated heatmap with Plotly (flipud to put high y-index at top = laser surface)
    frames = []
    for k in range(Nt):
        display_slice = np.flipud(interpolated_field[k, :, :])  # High y-index at top
        frames.append(go.Frame(
            data=go.Heatmap(
                z=display_slice, 
                colorscale=cmap, 
                zmin=t_min, 
                zmax=t_max,
                colorbar=dict(title=field_unit)
            ),
            name=f"t{k}"
        ))
    
    fig_anim = go.Figure(
        data=frames[0].data, 
        frames=frames,
        layout=go.Layout(
            title=f"{field_label} Evolution (y-axis: top = laser-heated surface)",
            xaxis_title="X index (scan direction)",
            yaxis_title="Y index (depth, top=surface)",
            updatemenus=[{
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]},
                    {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "type": "buttons",
                "x": 0.1,
                "y": 0
            }]
        )
    )
    
    # Slider for time navigation
    sliders = [{
        "active": t_idx,
        "steps": [
            {"method": "animate", "args": [[f"t{k}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}], "label": f"{k}"} 
            for k in range(Nt)
        ],
        "x": 0.1,
        "len": 0.9,
        "y": 0,
        "currentvalue": {"prefix": "Time step: ", "visible": True}
    }]
    fig_anim.update_layout(sliders=sliders)
    
    # Equal aspect ratio for physical realism
    fig_anim.update_yaxes(scaleanchor="x", scaleratio=1)
    
    st.plotly_chart(fig_anim, use_container_width=True)
    
    # === STATIC CONTOUR PLOT (High-resolution) ===
    with st.expander("📐 Static High-Resolution Contour Plot", expanded=False):
        slice_display = np.flipud(current_slice)  # For display: top=surface
        
        fig_static, ax = plt.subplots(figsize=(12, 8))
        
        # Contour fill with custom levels
        levels = np.linspace(slice_min, slice_max, 60)
        cont = ax.contourf(slice_display, levels=levels, cmap=cmap, origin='upper')
        
        # Add contour lines for clarity
        ax.contour(slice_display, levels=levels[::5], colors='white', linewidths=0.3, alpha=0.5)
        
        # Formatting
        ax.set_aspect('equal')  # Square cells for uniform grid
        ax.set_title(f"{field_label} @ Time Step {t_idx}", fontsize=14, fontweight='bold')
        ax.set_xlabel("X index (scan direction)", fontsize=11)
        ax.set_ylabel("Y index (depth, ↑ = surface)", fontsize=11)
        
        # Colorbar
        cbar = plt.colorbar(cont, ax=ax, label=field_unit)
        cbar.ax.tick_params(labelsize=9)
        
        # Grid and ticks
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_static)
    
    # === DOWNLOAD SECTION ===
    st.divider()
    st.subheader("💾 Download Interpolated Field")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        # NPZ download (preserves 3D structure)
        buffer_npz = io.BytesIO()
        metadata = {
            'P_target': p_target,
            'v_target_cm_s': v_target_cm_s,
            'TF_target': tf_target if field_type=='composition' else -1,
            'field_type': field_type,
            'interpolation_method': 'phase_aware_attention',
            'sigma_param': sigma_param,
            'sigma_comp': sigma_comp
        }
        np.savez_compressed(
            buffer_npz, 
            field=interpolated_field, 
            shape=np.array(shape),
            **metadata
        )
        buffer_npz.seek(0)
        
        suffix = "T" if field_type=='temperature' else "COMP"
        st.download_button(
            "📦 Download .npz (3D array + metadata)",
            buffer_npz,
            file_name=f"{suffix}_interp_P{int(p_target)}W_v{int(v_target_cm_s)}cm{'s' if field_type=='composition' else ''}_TF{tf_target if field_type=='composition' else 'NA'}.npz",
            mime="application/octet-stream",
            key="btn_npz"
        )
    
    with col_d2:
        # CSV download (flattened for spreadsheet analysis)
        flat_data = {
            'time_step': np.repeat(np.arange(Nt), Ny * Nx),
            'y_index': np.tile(np.repeat(np.arange(Ny), Nx), Nt),
            'x_index': np.tile(np.arange(Nx), Ny * Nt),
            field_label.split()[0]: interpolated_field.flatten()  # 'Temperature' or 'Composition'
        }
        flat_df = pd.DataFrame(flat_data)
        
        csv_buf = io.StringIO()
        flat_df.to_csv(csv_buf, index=False)
        
        st.download_button(
            "📊 Download .csv (flattened table)",
            csv_buf.getvalue(),
            file_name=f"{field_label.split()[0].lower()}_interp_P{int(p_target)}W_v{int(v_target_cm_s)}cm_s.csv",
            mime="text/csv",
            key="btn_csv"
        )
    
    # === ETALIQ INTEGRATION NOTE ===
    with st.expander("🔗 ETALIQ Phase Field Integration (Advanced)", expanded=False):
        st.markdown("""
        #### Using ETALIQ for Initial Condition Construction
        The interpolated field above uses process parameters (P, v) and composition index (TF) 
        to weight source simulations. For **complete physical consistency**, the initial composition 
        field should be constructed as:
        
        ```python
        # Pseudocode for initial condition setup
        eta_0 = load_etaliq_phase_field("ETALIQ/", target_filename)  # η(t=0, y, x)
        c_L = CTF_TABLE[TF_target]['liquid']  # [Co, Cr, Fe, Ni] for η=1
        c_S = CTF_TABLE[TF_target]['fcc']     # [Co, Cr, Fe, Ni] for η=0
        
        # Spatial construction of initial cTau tensor
        c_init = np.zeros((Ny, Nx, 4))  # 4 elements: Co,Cr,Fe,Ni
        for elem_idx in range(4):
            c_init[..., elem_idx] = eta_0[0,:,:] * c_L[elem_idx] + (1 - eta_0[0,:,:]) * c_S[elem_idx]
        ```
        
        This `c_init(y,x)` is then used as the initial condition for the phase-field evolution 
        equations that produce the composition solution fields in the COMP/ folder.
        
        🔹 **Note**: The current interpolation tool operates on the *evolved solution fields* 
        (TEMP/ or COMP/). The ETALIQ-based initial condition construction is a pre-processing 
        step that defines the simulation setup, not the interpolation target.
        """)
        
        # Optional: Load and display ETALIQ for target if available
        if field_type == 'composition':
            target_filename = f"p{int(p_target)}s{int(v_target_cm_s)}cTF{tf_target}.npy"
            eta_target = load_etaliq_phase_field(etaliq_folder, target_filename)
            
            if eta_target is not None:
                st.success(f"✅ Found ETALIQ phase field for target: `{target_filename}`")
                
                # Display initial phase field snapshot
                eta_0 = eta_target[0, :, :]  # t=0 slice
                fig_eta, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(np.flipud(eta_0), cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
                ax.set_title(f"Initial Phase Field η(t=0, y, x) — Top=Surface (η=1: Liquid)")
                ax.set_xlabel("X index")
                ax.set_ylabel("Y index (↑ = surface)")
                plt.colorbar(im, ax=ax, label="Phase fraction η")
                st.pyplot(fig_eta)
                
                st.caption("Color scale: 🔴 η=1 (LIQUID) ↔ 🔵 η=0 (FCC); purple = diffused interface (0<η<1)")
            else:
                st.warning(f"⚠️ ETALIQ file not found: `{os.path.join(etaliq_folder, target_filename)}`")

else:
    # No results yet - show guidance
    st.info("""
    ### 🎯 Next Steps
    1. **Load source simulations**: Ensure TEMP/ or COMP/ folder contains files matching `p*i*s*j*cTF*k.npy`
    2. **Configure target parameters**: Set desired P*, v*, and TF* (for composition fields) in sidebar
    3. **Run interpolation**: Click the green button to compute the attention-weighted field
    4. **Visualize & export**: Use animated plots, static contours, and download buttons
    
    🔍 **Tip**: Expand the "Detailed cTau Tensor Inspection" section above to verify initial composition parameters for each source.
    """)

# ============================================================================
# FOOTER: DOCUMENTATION & THEORY REFERENCE
# ============================================================================
st.divider()
with st.expander("📚 Theory Reference: Phase-Aware Attention Interpolation", expanded=False):
    st.markdown("""
    ### Mathematical Formulation
    
    #### 1. Parameter Normalization
    Process parameters are min-max normalized to [0,1]:
    ```
    P̃ = (P - P_min) / (P_max - P_min + ε)
    ṽ = (v - v_min) / (v_max - v_min + ε)
    ```
    
    #### 2. Process Parameter Attention
    Query/Key projections with multi-head attention:
    ```
    q = W_q · [P̃*, ṽ*]ᵀ ∈ ℝ^(H×d)
    k_i = W_k · [P̃_i, ṽ_i]ᵀ ∈ ℝ^(H×d)
    α_i = softmax( (k_i·q)/√d ) averaged over H heads
    ```
    
    #### 3. Gaussian Spatial Locality Prior
    Physics-inspired weighting favoring nearby parameters:
    ```
    s_i = exp(-‖[P̃_i,ṽ_i] - [P̃*,ṽ*]‖² / (2σ_param²))
    ŝ_i = s_i / Σ_j s_j
    ```
    
    #### 4. Composition Similarity (for composition fields)
    Gaussian similarity of initial mole fraction vectors:
    ```
    sim_i = 0.5·[exp(-‖c_i^L - c*^L‖²/(2σ_comp²)) + exp(-‖c_i^S - c*^S‖²/(2σ_comp²))]
    ```
    
    #### 5. Hybrid Weight Combination
    ```
    w_i = (α_i · ŝ_i · sim_i) / Σ_j (α_j · ŝ_j · sim_j)
    ```
    
    #### 6. Field Interpolation
    ```
    T*(t,y,x) = Σ_i w_i · T_i(t,y,x)   [for temperature]
    x_m*(t,y,x) = Σ_i w_i · x_m,i(t,y,x)   [for composition element m]
    ```
    
    ### Key Distinctions
    - **cTFk in filename**: Encodes INITIAL mole fractions only (t=0 parameters)
    - **ETALIQ container**: Provides η(t,y,x) to spatially construct c_init(y,x) at t=0
    - **Solution fields**: Evolved fields T(t,y,x) or x_m(t,y,x) stored in TEMP/ or COMP/
    - **Interpolation**: Operates on solution fields using (P,v) or (P,v,TF_idx) as keys
    
    ### References
    - Phase-field modeling of HEAs: Subramanian et al., Acta Materialia (2024)
    - Attention mechanisms for scientific ML: Vaswani et al., NeurIPS (2017)
    - Gaussian process priors for parameter interpolation: Rasmussen & Williams (2006)
    """)

st.caption("🔬 Phase-Field Aware Attention Interpolation Tool v2.0 | Co-Cr-Fe-Ni High-Entropy Alloy Laser Processing | Dr. Subramanian Research Group")
