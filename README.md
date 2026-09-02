# ProstateOrdinalGPClassification with OmegaConf

This is an enhanced version of the ProstateOrdinalGPClassification project that uses **OmegaConf** and **Hydra** for configuration management, making it much easier to run experiments, perform hyperparameter sweeps, and manage different configurations.

## Key Improvements

- 🔧 **Structured Configuration**: YAML-based configuration files instead of command-line arguments
- 🚀 **Easy Hyperparameter Sweeps**: Built-in support for running multiple configurations
- 📁 **Organized Results**: Automatic output directory management with timestamps
- 🔄 **Configuration Inheritance**: Modular configs for datasets, models, and experiments
- 📊 **Better Reproducibility**: All configurations are logged and saved with results

## Installation

### Option 1: Using Conda (Recommended)

```bash
cd ProstateOrdinalGPClassification_OmegaConf

# Automatic setup
./setup_environment.sh

# Manual setup
conda env create -f environment.yaml
conda activate prostate-ordinal-gp
```

### Option 2: Using pip

```bash
cd ProstateOrdinalGPClassification_OmegaConf
pip install -r requirements.txt
```

### Verify Installation

```bash
# Activate environment (if using conda)
conda activate prostate-ordinal-gp

# Test the configuration system
python validate_config.py
```

## Configuration Structure

```
configs/
├── config.yaml              # Main configuration file
├── dataset/                  # Dataset configurations
│   ├── sicap.yaml
│   ├── crowdgleason.yaml
│   ├── sicap_uni_full.yaml
│   └── crowdgleason_uni_full.yaml
├── model/                    # Model configurations
│   ├── ordinal_svgp.yaml
│   ├── ordinal_dsvi_dgp.yaml
│   └── svgp_multiclass.yaml
└── experiment/               # Experiment configurations
    ├── base.yaml
    └── ablation.yaml
```

## Usage Examples

### 1. Basic Usage

Run with default configuration:
```bash
python src/experiments/main.py
```

### 2. Override Parameters

Override specific parameters from command line:
```bash
# Change dataset and seed
python src/experiments/main.py dataset=crowdgleason seed=2

# Modify training parameters
python src/experiments/main.py training.learning_rate=0.05 training.epochs=5000

# Use different model
python src/experiments/main.py model=ordinal_dsvi_dgp
```

### 3. Hyperparameter Sweeps

Run multiple configurations with Hydra multirun:
```bash
# Simple sweep over seeds and learning rates
python src/experiments/main.py -m seed=0,1,2,3,4 training.learning_rate=0.01,0.1

# Complex sweep over multiple parameters
python src/experiments/main.py -m \
    dataset=sicap,crowdgleason \
    model=ordinal_svgp \
    seed=0,1,2 \
    training.learning_rate=0.01,0.05,0.1 \
    model.inducing_points.num_inducing=200,500
```

## Pre-built Scripts

Several pre-built scripts are available in `bash_scripts/`:

Run the added ordinal baselines with the standard experiment pipeline:
```bash
PYTHON_BIN=/home/fjaviersaezm/.conda/envs/GPFLOW/bin/python \
MODELS="sord beta poe core drf corf" \
DATASETS="sicap crowdgleason" \
FEATURES="UNI2 PHIKONV2 VIRCHOW VIRCHOW2 UNI GIGAPATH CONCHV15" \
SEEDS="0 1 2 3 4" \
bash bash_scripts/run_ordinal_baselines.sh
```

For a short SORD trial, reduce the sweep and epoch count:
```bash
PYTHON_BIN=/home/fjaviersaezm/.conda/envs/GPFLOW/bin/python \
MODELS="sord" DATASETS="sicap" FEATURES="UNI" SEEDS="0" \
EPOCHS=20 BATCH_SIZE=128 MAX_PROCESSES=1 \
bash bash_scripts/run_ordinal_baselines.sh
```

Evaluate the 35 saved CrowdGleason OrdinalSVGP checkpoints while retaining
the learned likelihood variance, and add namespaced metrics to the original
W&B run summaries:
```bash
LD_PRELOAD=/home/fjaviersaezm/.conda/envs/ordinalgp/lib/libstdc++.so.6 \
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
/home/fjaviersaezm/.conda/envs/ordinalgp/bin/python \
  src/experiments/evaluate_crowdgleason_likelihood_variance.py \
  --update-wandb
```

The original `test/*` metrics are preserved. New values use the
`test_with_likelihood_variance/*` prefix and are also exported to
`results/crowdgleason_ordinal_svgp_likelihood_variance_on.csv`.

1. **Ordinal SVGP experiments**:
   ```bash
   ./bash_scripts/run_ordinal_svgp_hydra.sh
   ```

2. **Ordinal DSVI-DGP ablation study**:
   ```bash
   ./bash_scripts/run_ordinal_dsvi_dgp_hydra.sh
   ```

3. **Demo of all features**:
   ```bash
   ./bash_scripts/demo_omegaconf.sh
   ```

## Configuration Management

The system uses modular YAML configuration files that can be easily combined and overridden. This makes it much easier to:

- Run hyperparameter sweeps
- Organize experiment results
- Reproduce experiments
- Share configurations

## Migration from Original Version

The new system is backward compatible with the original argparse-based code. All existing functionality is preserved while adding the benefits of structured configuration management.

For detailed documentation, see the full README with all configuration options and examples.
