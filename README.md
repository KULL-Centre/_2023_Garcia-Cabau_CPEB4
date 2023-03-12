
# Molecular simulations of the N-terminal domain of CPEB4

This repository contains Python code, [Jupyter](http://jupyter.org) Notebooks, and simulation data for reproducing the simulation results of the manuscript _Kinetic stabilization of translation-repression condensates by a neuron-specific microexon_

### Layout

- `analyses.ipynb` Jupyter Notebook to analyze all the simulation data and generate plots
- `direct-coexistence/` Data and Python code related to multi-chain simulations of the CALVADOS model in slab geometry. Simulations are performed using [openMM](https://openmm.org/) v7.5
- `multimers/` Data and Python code related to multi-chain simulations of multimer formation. Simulations are performed using [openMM](https://openmm.org/) v7.5 and analysed using [OVITO Basic](https://www.ovito.org/) v3.7

### Usage

To open the Notebooks, install [Miniconda](https://conda.io/miniconda.html) and make sure all required packages are installed by issuing the following terminal commands

```bash
    conda env create -f environment.yml
    source activate cpeb4
    jupyter-notebook
```
