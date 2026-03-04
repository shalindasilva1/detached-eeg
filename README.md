# Detach-Rocket Ensemble for EEG Classification

This repository contains an optimized implementation for classifying raw MEG/EEG data using the Detach-Rocket Ensemble algorithm. This project specifically replicates the methodology from the paper ["Classification of Raw MEG/EEG Data with Detach-Rocket Ensemble"](https://arxiv.org/abs/2408.02760).

## Experimental Replication Results

We successfully replicated the Section 5.2 experiment (AD vs. CN classification) on a 65-subject cohort.

* **Target Accuracy (from paper)**: 86.15%
* **Replication Accuracy (Subject-Level)**: **84.62%** (55/65 subjects correctly classified)

> [!NOTE]
> The current setup uses a 10-model Detach-Rocket ensemble trained over a full Leave-One-Subject-Out (LOSO) cross-validation loop. Due to our implementation's **pre-transformation strategy**, we extract all 10,000+ kernels on the GPU upfront. This optimizes execution time.

## Installation

We recommend using Conda to manage your python environment.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd detached-eeg
   ```

2. **Create the Conda Environment:**
   If you rely on PyTorch and CUDA (highly recommended for `detach_rocket`), create a fresh environment:
   ```bash
   conda create -n eeg-rocket python=3.10
   conda activate eeg-rocket
   ```

3. **Install Dependencies:**
   Install PyTorch specifically for your CUDA version (e.g., CUDA 11.8):
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
   Then install the rest of the requirements, including the `detach-rocket` repository:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Configuration (`config.yml`)
Ensure your paths and experimental settings are defined in `config.yml`. By default, it expects:
* `data.resting`: Path to BIDS formatted resting-state EEG dataset
* `experiment.n_splits`: Default 5
* `model.params.num_models`: Default 10
* `model.params.num_kernels`: Default 10000

### 2. Running the Pipeline
Execute the main classification script. This will automatically check for CUDA availability, load the dataset, run the pre-transformations, and execute the LOSO loop.

```bash
python src/pipeline.py
```
*Results will save to `results/loso_results.json`.*

### 3. Visualizing Results
After a complete run, generate performance plots (Confusion Matrix and Trial-Level Accuracy):

```bash
python src/visualize_results.py
```
*Figures are saved to the `results/figures/` directory.*

## Project Structure
* `src/pipeline.py`: Main execution script for the classification pipeline.
* `src/visualize_results.py`: Chart generation logic.
* `results/`: Contains the generated JSON reports and output figures.
* `config.yml`: Core configuration file for paths and hyper-parameters.
* `requirements.txt`: Python dependencies.
