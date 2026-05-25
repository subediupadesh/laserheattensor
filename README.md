# Attention-Regularized Multicomponent Phase-Field Surrogate for Laser Processing of CoCrFeNi Alloy

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-red.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-ML%20Surrogate-orange.svg)]()
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visualization-purple.svg)]()
[![Phase--Field](https://img.shields.io/badge/Phase--Field-Multiphysics-green.svg)]()
[![License](https://img.shields.io/badge/License-To%20be%20updated-lightgrey.svg)]()

This repository contains the computational workflow, thermodynamic data processing, phase-field simulation post-processing, machine-learning tensor organization, and attention-regularized surrogate model developed for the study:

## Paper

**Attention-Regularized Multicomponent Phase-Field Surrogate for Laser Processing of CoCrFeNi Alloy**

## Authors

[Upadesh Subedi](https://www.linkedin.com/in/upadesh-s-0b321a15b/),  
[Nele Moelans](https://www.linkedin.com/in/nele-moelans-57b1731/),  
[Tomasz Tański](https://www.linkedin.com/in/tomasz-tanski-888bb266/),  
[Anil Kunwar](https://www.linkedin.com/in/anil-kunwar-9ba81653/)

---

## Overview

This project develops an attention-regularized surrogate framework for predicting laser-induced thermal, phase, velocity, and compositional fields in a multicomponent CoCrFeNi alloy system.

The framework combines:

- Thermodynamic tensor construction for multicomponent phase stability analysis.
- Multiphysics phase-field simulation data from laser processing cases.
- Post-processing and tensor preparation of simulation outputs.
- Attention-based interpolation across laser processing parameters.
- Interactive visualization of surrogate predictions.
- Interpretation of source-simulation contributions through attention-weight analysis.

The surrogate model is designed to interpolate high-fidelity phase-field simulation outputs across laser power and scan-speed conditions while preserving physically meaningful relationships between temperature evolution, liquid-phase formation, meltpool velocity, and composition redistribution.

---

## Scientific Motivation

Laser processing of multicomponent alloys involves tightly coupled thermal, kinetic, and thermodynamic phenomena. Direct phase-field simulations can provide high-fidelity insight into these processes, but they are computationally expensive and not suitable for rapid exploration of large processing windows.

This repository addresses that limitation by building a surrogate framework that learns from phase-field simulation tensors and predicts spatiotemporal fields for unseen laser-processing conditions.

The central idea is to use attention-regularized interpolation in process-parameter space. Instead of treating each simulation independently, the model evaluates how strongly each available source simulation should contribute to a target processing condition.

The project focuses on CoCrFeNi alloy laser processing, with particular emphasis on:

- Temperature field evolution.
- LIQUID/FCC phase evolution.
- Meltpool morphology.
- Velocity prediction inside the liquid region.
- Composition-dependent thermodynamic tensor representation.
- Processing-condition diagnosis.
- Attention-weight decomposition.
- Query-key heatmap visualization.
- Source-contribution analysis.

---

## Repository Structure

```text
.
├── 00_Thermodynamic_Data_Tensor
│   ├── csv_files
│   ├── figures
│   └── tdb_files
├── 1_Simulation_Results
├── 2_Post_Processing
│   ├── Data_Organization
│   ├── figures
│   └── quantification
├── 3_ML_Model
│   ├── MLDATA
│   │   ├── COMP
│   │   ├── ETALIQ
│   │   ├── SIM_TIME
│   │   ├── TEMP
│   │   └── VEL
│   └── theory
├── alloylpbf_informatics
└── README.md
```

---

## Project Modules

### 1. Thermodynamic Data Tensor

The `00_Thermodynamic_Data_Tensor` directory contains the thermodynamic foundation of the project.

This module includes:

- Gibbs free-energy data across temperature.
- Thermodynamic database files.
- Thermodynamic tensor construction tools.
- Composition-temperature visualization scripts.
- Interface driving-force analysis.
- 8D parallel-coordinate plotting.
- Free-energy landscape visualization.
- Tetrakaidecahedron grain visualization.

The purpose of this module is to connect alloy composition, temperature, Gibbs free energy, and phase stability into a structured thermodynamic tensor representation.

This thermodynamic tensor serves as the physics-informed background for understanding phase stability and phase transformation tendencies in the CoCrFeNi alloy system.

---

### 2. Simulation Results

The `1_Simulation_Results` directory is reserved for high-fidelity phase-field simulation results obtained from laser processing simulations of the CoCrFeNi alloy.

These simulations serve as the physics-based source data for the surrogate model.

Typical simulation outputs include:

- Temperature field.
- LIQUID phase order parameter.
- FCC phase order parameter.
- Velocity field.
- Composition fields.
- Simulation time history.

These data are later post-processed and converted into machine-learning-ready tensor formats.

---

### 3. Post-Processing

The `2_Post_Processing` directory contains tools for organizing, reading, quantifying, and visualizing phase-field simulation outputs.

This module supports:

- Reading simulation outputs.
- Extracting spatiotemporal fields.
- Converting simulation data into NumPy tensor format.
- Preparing data for machine-learning prediction.
- Quantifying phase evolution.
- Quantifying meltpool area.
- Reading simulation time arrays.
- Producing figures for thermodynamic and surrogate interpretation.

The post-processing workflow bridges the raw simulation outputs and the surrogate-model input format.

---

### 4. Machine-Learning Surrogate Model

The `3_ML_Model` directory contains the attention-regularized surrogate model and the organized machine-learning data.

This module is the core of the project.

The surrogate predicts spatiotemporal fields for target laser processing conditions using available phase-field simulations as source cases.

The machine-learning data are organized as:

```text
MLDATA
├── COMP
├── ETALIQ
├── SIM_TIME
├── TEMP
└── VEL
```

These directories represent:

| Directory | Meaning |
|---|---|
| `COMP` | Composition field tensors |
| `ETALIQ` | LIQUID phase-field tensors |
| `SIM_TIME` | Simulation time arrays |
| `TEMP` | Temperature field tensors |
| `VEL` | Velocity field tensors |

The surrogate can predict:

- Temperature field: `T(t, y, x)`.
- Liquid phase field: `ηLIQ(t, y, x)`.
- Velocity field: `v(t, y, x)`.
- Composition field: `c(t, y, x)`.

The attention mechanism combines learned source importance with process-parameter locality.

---

### 5. Alloy LPBF Informatics

The `alloylpbf_informatics` directory contains auxiliary tools and descriptors related to laser powder bed fusion processing windows.

This module supports broader interpretation of laser-processing conditions through:

- Processing-window descriptors.
- Key terminology.
- Process-regime information.
- Auxiliary data for alloy LPBF interpretation.

Although the main surrogate model focuses on phase-field simulation tensors, this directory helps contextualize the processing conditions from an alloy manufacturing perspective.

---

## Core Workflow

The project follows the workflow below:

```text
Thermodynamic Database / Gibbs Energy Data
                │
                ▼
Thermodynamic Tensor Construction
                │
                ▼
Phase-Field Laser Processing Simulations
                │
                ▼
Post-Processing and Tensor Organization
                │
                ▼
Attention-Regularized Surrogate Prediction
                │
                ▼
Interactive Visualization and Processing Diagnosis
```

---

## Main Capabilities

This repository enables:

- Construction of thermodynamic tensors for CoCrFeNi alloy.
- Visualization of Gibbs free-energy landscapes.
- Analysis of temperature-composition-phase relationships.
- Interface driving-force quantification.
- Organization of phase-field simulation outputs into machine-learning tensors.
- Attention-based interpolation of unseen laser-processing conditions.
- Prediction of temperature, LIQUID phase, velocity, and composition fields.
- Plotly-based interactive animation of surrogate-predicted fields.
- Quantification of meltpool area from predicted LIQUID phase fields.
- Laser-processing diagnosis based on melting and boiling thresholds.
- Interpretation of surrogate behavior through attention-weight decomposition.
- Query-key heatmap visualization for attention analysis.
- Source-contribution mapping across laser power and scan speed.

---

## Attention-Regularized Surrogate Concept

For a target laser-processing condition:

```text
P* = target laser power
v* = target scan speed
```

the model compares the target condition with available source simulations.

Each source simulation contributes to the final surrogate prediction through a hybrid weight:

```text
Hybrid weight = Attention weight × Gaussian locality weight
```

where:

- The attention weight is obtained from transformer-inspired query-key projection of processing parameters.
- The Gaussian locality weight favors source simulations closer to the target point in process-parameter space.
- The final hybrid weight is normalized across all source simulations.
- The predicted field is reconstructed as a weighted combination of source simulation tensors.

For a field variable `F`, the surrogate prediction can be interpreted as:

```text
F*(t, y, x) = Σ wi Fi(t, y, x)
```

where:

- `F*` is the predicted field at the target condition.
- `Fi` is the field from the `i-th` source simulation.
- `wi` is the hybrid attention-locality weight assigned to that source.

This makes the surrogate interpretable because the contribution of each source simulation can be inspected directly.

---

## Predicted Physical Fields

The surrogate framework can be applied to multiple phase-field outputs.

### Temperature Prediction

The temperature prediction module estimates the spatiotemporal temperature field:

```text
T(t, y, x)
```

This output is used to analyze:

- Thermal history.
- Peak temperature.
- Heat-affected region.
- Meltpool formation tendency.
- Undermelting and overheating regimes.

### ETALIQ Prediction

The ETALIQ prediction module estimates the LIQUID phase order parameter:

```text
ηLIQ(t, y, x)
```

This output is used to analyze:

- Meltpool morphology.
- Liquid-phase area.
- Liquid-solid interface evolution.
- Time-dependent meltpool growth and shrinkage.

### Velocity Prediction

The velocity prediction module estimates meltpool flow behavior:

```text
v(t, y, x)
```

Velocity interpretation is physically meaningful primarily inside the LIQUID region. Therefore, velocity fields can be filtered using the predicted LIQUID phase field so that velocity is shown only inside the meltpool region.

### Composition Prediction

The composition prediction module estimates spatial and temporal compositional redistribution:

```text
c(t, y, x)
```

This output is used to connect thermodynamic driving force, phase stability, and multicomponent redistribution during laser processing.

---

## Laser-Processing Diagnosis

The surrogate-predicted temperature field can be used to diagnose the quality of laser processing.

For the CoCrFeNi alloy system used in this project:

```text
Melting temperature: 1828.52 K
Boiling temperature: 3116.40 K
```

The predicted thermal condition can be interpreted as:

| Temperature Range | Diagnosis | Physical Meaning |
|---|---|---|
| `T < 1828.52 K` | Undermelting | The alloy does not reach the melting point. |
| `1828.52 K ≤ T < 3116.40 K` | Stable melting | The alloy forms a liquid meltpool without reaching the boiling point. |
| `T ≥ 3116.40 K` | Possible ablation / vaporization | Excessive heating may cause evaporation, recoil pressure, or unstable meltpool behavior. |

This diagnosis is useful for identifying whether a selected combination of laser power and scan speed produces insufficient melting, stable meltpool formation, or excessive overheating.

---

## Data Format

The machine-learning tensors are stored as NumPy arrays.

Typical tensor dimensions are:

```text
time × y-coordinate × x-coordinate
```

For scalar fields such as temperature, LIQUID phase, and velocity:

```text
F.shape = (Nt, Ny, Nx)
```

For composition fields with multiple components, the format may be:

```text
COMP.shape = (Nt, Nc, Ny, Nx)
```

or another channel-based format depending on preprocessing.

Here:

| Symbol | Meaning |
|---|---|
| `Nt` | Number of time steps |
| `Ny` | Number of spatial points along y-direction |
| `Nx` | Number of spatial points along x-direction |
| `Nc` | Number of composition channels |

---

## Processing Parameters

The surrogate uses laser-processing parameters as input descriptors.

The primary descriptors are:

| Parameter | Meaning |
|---|---|
| `P` | Laser power |
| `v` | Scan speed |

The source simulations are identified by combinations of power and scan speed. The surrogate predicts the output field for a target combination of these parameters.

---

## Visualization Features

The project supports both static and interactive visualization.

Key visualization features include:

- 8D parallel-coordinate thermodynamic tensor plots.
- Gibbs free-energy surface plots.
- Interface driving-force visualization.
- Tetrakaidecahedron grain visualization.
- Temperature-field animation.
- LIQUID phase-field animation.
- Velocity-field animation.
- Composition-field maps.
- Meltpool area evolution.
- Laser-processing diagnosis maps.
- Attention-weight decomposition.
- Query-key heatmaps.
- Chord-style source-contribution diagrams.
- Process-composition contribution maps.
- Transparent-background figure export for publication use.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Create a clean Python environment:

```bash
conda create -n cocrfeni_surrogate python=3.10
conda activate cocrfeni_surrogate
```

Install the required packages:

```bash
pip install numpy pandas matplotlib plotly streamlit torch scipy scikit-learn seaborn
```

Depending on the specific module being used, additional packages may be required.

For thermodynamic database or phase-diagram related analysis, packages such as `pycalphad` may also be needed.

---

## Running the Surrogate App

From the repository root, move into the machine-learning directory:

```bash
cd 3_ML_Model
```

Run the Streamlit app:

```bash
streamlit run Attention_Regularized_Surrogate.py
```

The interactive interface allows the user to:

- Select target laser power.
- Select target scan speed.
- Adjust Gaussian locality regularization.
- Select the predicted field.
- Animate the surrogate result.
- Inspect area evolution.
- Inspect processing diagnosis.
- Visualize source-simulation contributions.
- Analyze attention weights.

---

## Suggested Usage

A typical workflow is:

1. Generate phase-field simulation outputs for selected laser-processing conditions.
2. Post-process the simulation outputs into structured tensor data.
3. Store temperature tensors in `MLDATA/TEMP`.
4. Store LIQUID phase tensors in `MLDATA/ETALIQ`.
5. Store velocity tensors in `MLDATA/VEL`.
6. Store composition tensors in `MLDATA/COMP`.
7. Store simulation time arrays in `MLDATA/SIM_TIME`.
8. Launch the surrogate app.
9. Select a target power and scan speed.
10. Run the surrogate prediction.
11. Visualize temperature, LIQUID phase, velocity, and composition fields.
12. Analyze meltpool area and laser-processing diagnosis.
13. Inspect attention weights and source-simulation contributions.

---

## Example Surrogate Interpretation

For a target condition located between multiple source simulations, the model assigns different hybrid weights to nearby source cases.

A physically reasonable interpolation should show:

- Stronger contribution from simulations close to the target power and scan speed.
- Smooth transition in predicted temperature field.
- Consistent LIQUID phase morphology with the predicted temperature.
- Higher meltpool area for higher power or lower scan speed.
- Lower meltpool area for lower power or higher scan speed.
- Possible ablation warning when predicted temperature exceeds the boiling threshold.
- Undermelting warning when predicted temperature remains below the melting threshold.

---

## Research Significance

This repository contributes toward physics-aware digital twins for laser processing of multicomponent alloys.

Instead of replacing phase-field modeling with a black-box machine-learning model, the framework uses high-fidelity phase-field simulations as the knowledge source and builds an interpretable surrogate layer on top of them.

The approach is useful for:

- Rapid exploration of processing windows.
- Reducing the need for repeated expensive phase-field simulations.
- Understanding power-scan speed effects on meltpool evolution.
- Linking thermodynamic stability with microstructure prediction.
- Interpreting surrogate predictions through source-simulation contribution.
- Developing interactive tools for process-microstructure analysis.
- Supporting future digital-twin frameworks for laser-based alloy processing.

---

## Current Status

This repository is organized as a research-codebase accompanying the paper:

```text
Attention-Regularized Multicomponent Phase-Field Surrogate for Laser Processing of CoCrFeNi Alloy
```

The codebase includes thermodynamic visualization, post-processing, machine-learning tensor organization, and interactive surrogate-model tools.

Some paths, filenames, dependency versions, and execution commands may need to be adjusted depending on the local machine setup.

---

## Citation

If you use this repository, workflow, or data organization concept, please cite the associated paper:

```bibtex
@article{Subedi2026b,
  title   = {Attention-Regularized Multicomponent Phase-Field Surrogate for Laser Processing of CoCrFeNi Alloy},
  author  = {Subedi, Upadesh and Moelans, Nele and Tański, Tomasz and Kunwar, Anil},
  journal = {To be updated},
  year    = {2026},
  doi     = {To be updated}
}
```

---

## License

The license for this repository will be updated upon publication or public release.

Until then, please contact the authors before reusing, redistributing, or modifying the repository content.

---

## Contact

For questions, collaboration, or discussion related to this project, please contact:

**Upadesh Subedi**  
PhD Researcher, Computational Materials Science  
Silesian University of Technology  

LinkedIn: [upadesh-s-0b321a15b](https://www.linkedin.com/in/upadesh-s-0b321a15b/)
