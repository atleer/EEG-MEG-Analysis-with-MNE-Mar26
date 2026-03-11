# Workshop on EEG and MEG Analysis with MNE-Python

This repo contains the materials for a 3-day online workshop on analyzing EEG and MEG data in Python using mainly the MNE package.

## Schedule

- When: 11.03.2026 - 13.03.2026 @ 9:30 - 17:00 (lunch break from 12:30 to 13:45)
- Where: online (zoom)

### Day 1: From Raw Recordings to Evoked Responses

1. Working with raw recordings
2. Simulating EEG using a forward model
3. Epoching recordings to compute ERPs
4. Statistical analysis of ERPs

### Day 2: Separating Signals in Space and Time

1. Spatial filtering with PCA and ICA
2. Temporal filtering with FIR filters
3. Time-frequency analysis
4. Phase-based connectivity

### Day 3: Linear Models for Neural Signals

1. Linear regression and regularization
2. Estimating time-varying relationships
3. Modeling brain responses to natural speech
4. Demo: linear models on widefield imaging data


## Setup

### Installation

1. Install [pixi](https://pixi.prefix.dev/latest/installation/) and [git](https://git-scm.com/install/)
2. Clone this repository: `git clone https://github.com/ibehave-ibots/EEG-MEG-Analysis-with-MNE-Mar26.git`
3. Move to the cloned directory (`cd EEG-MEG-Analysis-with-MNE-Mar26`) and run `pixi install`

### Selecting the Kernel

When you open a notebook in VSCode click "Select Kernel" > "Python Environments" and select the environment called **default** (Python Version 3.12.13).

If this environment does not show up do `pixi run install-kernel` and then click "Select Kernel" > "Jupyter Kernel" and select **ibots-eeg** (you may need to refresh the list so it shows up)
