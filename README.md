# LaserHEATTENsor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Marimo](https://img.shields.io/badge/Marimo-Interactive%20Apps-orange.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-AI%20Surrogate-red.svg)]()
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-purple.svg)]()
[![Phase--Field](https://img.shields.io/badge/Phase--Field-CoCrFeNi-green.svg)]()

**LaserHEATTENsor** is a research codebase for connecting thermodynamic tensor representations, non-isothermal multicomponent phase-field simulations, and attention-regularized AI surrogates for laser additive manufacturing of CoCrFeNi alloy.

🌐 **Interactive web app:** [LaserHEATTENsor](https://subediupadesh.github.io/laserheattensor/)

---

## Manuscript

**From Thermodynamic Tensors to Attention-Regularized Surrogates: A Multicomponent Phase-Field Framework for Laser Additive Manufacturing of CoCrFeNi Alloy**

**Authors:** Upadesh Subedi, Nele Moelans, Tomasz Tański, Anil Kunwar

---

## Overview

This repository supports a physics-to-surrogate workflow for laser processing of CoCrFeNi alloy. The project starts from CALPHAD-derived Gibbs free-energy data, organizes the information as a high-dimensional thermodynamic data tensor, extracts factor-matrix structure using tensor decomposition, and formulates a computationally efficient quadratic thermodynamic approximation for phase-field simulations.

The resulting simulation tensors are then used to build an attention-regularized surrogate model that predicts temperature and LIQUID phase evolution for unseen laser processing conditions. The surrogate combines cross-attention, Gaussian locality, and composition-aware weighting to preserve physically meaningful interpolation across laser power, scan speed, and phase-conditioned alloy chemistry.

---

## Core Workflow

```text
CALPHAD / Gibbs-energy data
        ↓
Thermodynamic Data Tensor (TDT)
        ↓
CPD factor matrices and tensor visualization
        ↓
Quadratic thermodynamic approximation for phase-field modeling
        ↓
Non-isothermal CoCrFeNi laser phase-field simulations
        ↓
Tensorized temperature, phase, velocity, composition, and time data
        ↓
Attention-regularized AI surrogate
        ↓
Interactive visualization and ablation assessment
```

---

## Main Features

- Thermodynamic data tensor construction for CoCrFeNi alloy.
- Interactive 8D composition-temperature-Gibbs energy visualization.
- CPD-based factor-matrix analysis for composition and temperature modes.
- Quadratic approximation of reconstructed thermodynamic tensor data for phase-field use.
- Non-isothermal multicomponent phase-field modeling of LIQUID/FCC evolution.
- Meltpool flow modeling with Marangoni effect.
- Machine-learning-ready tensor organization of simulation outputs.
- Attention-regularized surrogate prediction for temperature and phase fields.
- Ablation and undermelting assessment using surrogate-predicted data.
- Browser-based interactive visualization through the LaserHEATTENsor app.

---

## Repository Structure

```text
.
├── 0_Thermodynamic_Data_Tensor
├── 1_Simulation_Results
├── 2_Post_Processing
├── 3_ML_Model
├── 4_Ablation_Assessment
├── alloylpbf_informatics
├── test_deployments
├── README.md
└── requirements.txt
```

### Directory Summary

| Directory | Purpose |
|---|---|
| `0_Thermodynamic_Data_Tensor` | Thermodynamic data processing, tensor construction, factor matrices, Gibbs-energy visualization, and Marimo apps. |
| `1_Simulation_Results` | Phase-field simulation outputs for laser-processed CoCrFeNi cases. |
| `2_Post_Processing` | Data extraction, organization, quantification, and figure generation from simulation results. |
| `3_ML_Model` | Attention-regularized surrogate model, tensorized ML data, prediction tools, and theory notes. |
| `4_Ablation_Assessment` | Surrogate-based thermal diagnosis, undermelting/ablation analysis, and predicted-field assessment. |
| `alloylpbf_informatics` | Auxiliary LPBF processing-window descriptors and key-term files. |
| `test_deployments` | GitHub Pages and Marimo deployment test files. |

---

## Interactive Apps

The project includes browser-ready Marimo visualizations hosted through GitHub Pages:

🔗 **Main app page:** [https://subediupadesh.github.io/laserheattensor/](https://subediupadesh.github.io/laserheattensor/)

Current app modules include thermodynamic tensor visualization, multi-ring chart visualization, parallel-coordinate Gibbs-energy plots, and surrogate/ablation analysis tools.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/subediupadesh/laserheattensor.git
cd laserheattensor
```

Create and activate a clean environment:

```bash
conda create -n laserheattensor python=3.10
conda activate laserheattensor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For individual modules, additional local requirements may be present inside subdirectories.

---

## Running Locally

Run a Marimo app, for example:

```bash
marimo run 0_Thermodynamic_Data_Tensor/8D_Parallel_Plot_Marimo.py
```

or:

```bash
marimo run 0_Thermodynamic_Data_Tensor/Sunburst_Multiring_Marimo.py
```

Run the surrogate model script from the ML directory:

```bash
cd 3_ML_Model
python Attention_Regularized_Surrogate.py
```

---

## Data Notes

Large simulation tensors are stored as NumPy arrays, typically following:

```text
field(t, y, x)
```

The main ML data folders include:

```text
MLDATA/COMP
MLDATA/ETALIQ
MLDATA/SIM_TIME
MLDATA/TEMP
MLDATA/VEL
```

These represent composition, LIQUID phase-field, simulation time, temperature, and velocity tensors used by the surrogate framework.

---

## Citation

If you use this repository, workflow, or visualization tools, please cite the associated manuscript:

```bibtex
@article{Subedi2026LaserHEATTENsor,
  title   = {From Thermodynamic Tensors to Attention-Regularized Surrogates: A Multicomponent Phase-Field Framework for Laser Additive Manufacturing of CoCrFeNi Alloy},
  author  = {Subedi, Upadesh and Moelans, Nele and Tański, Tomasz and Kunwar, Anil},
  journal = {Submitted},
  year    = {2026}
}
```

---

## License

The license will be updated upon publication or public release. Until then, please contact the authors before reusing, redistributing, or modifying the repository content.

---

## Contact

**Upadesh Subedi**  
Computational Materials Science  
Silesian University of Technology  

LinkedIn: [Upadesh Subedi](https://www.linkedin.com/in/upadesh-s-0b321a15b/)
