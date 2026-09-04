


# LDIF: Latent Dual Interaction Flow

**LDIF** is a PyTorch implementation of the Latent Dual Interaction Flow. It is a continuous-time neural architecture with response-conditioned gating and adaptive spectrum pruning.

## Installation

```bash
pip install ldif-model==0.1.0
````

## Quick Start

### Static (tabular) data

LDIF can be implemented in Python for static data using the following example:

```python
import torch
from ldif_model import LDIFStatic

model = LDIFStatic(
    input_dim=128,
    output_dim=1,
    task_type='binary_classification',
    num_layers=2,
    k_max=16,
    r=8,
    gamma=0.01
)

x = torch.randn(32, 128)
logits = model(x)   # shape (32,)
```

For regression, use `task_type='regression'`; for multiclass classification, use `task_type='multiclass'` and set `output_dim` accordingly.

### Sequential (time-series) data

LDIF can be implemented in Python for sequential (temporal) data using the following example:

```python
import torch
from ldif_model import LDIFSequential

model = LDIFSequential(
    input_dim=8,
    output_dim=1,
    task_type='regression',
    num_layers=2,
    k_max=16,
    r=8,
    gamma=0.2
)

# x shape: (batch, sequence_length, features)
x = torch.randn(16, 50, 8)
out = model(x)   # shape (16,)
```

## Default Hyperparameters

The default hyperparameters for both `LDIFStatic` and `LDIFSequential` are explained here.

### LDIFStatic

LDIFStatic is tuned for static tabular datasets (e.g., regression or classification on feature vectors). The defaults are chosen to balance capacity and stability for moderate-sized data (up to ~10k samples):

* `num_layers=2`: Two stacked LDIF blocks provide sufficient depth to capture non-linear interactions without overfitting.
* `k_max=16`: The adaptive spectrum has a maximum rank of 16, which is ample for most tabular problems; the L1 penalty will prune unnecessary components.
* `r=8`: The gate projections use a rank of 8, offering a good trade-off between routing expressiveness and parameter efficiency.
* `gamma=0.01`: A mild decay coefficient prevents unbounded growth of the hidden state while allowing long-term information retention.
* `input_scale=2.0`: A relatively strong input injection boosts the signal from the raw features, helping the model quickly adapt to the data.
* `z_init_mean=1.0`: The spectrum logits start with a mean of 1.0, corresponding to sigmoid values around 0.73. This encourages most components to be initially active, giving the model full capacity from the start.
* `z_init_std=0.01`: Very tight initialization ensures that the spectrum does not vary wildly early in training, promoting stable gradient flow.

#### L1 Spectrum Penalty (μ)

`mu` is a hyperparameter representing the L1 penalty and is explained in the paper. The value of `mu` is dependent on the number of training samples.

It is preferred to define `N_train` as follows:

```python
N_train = len(train_loader.dataset)   # if using DataLoader
```

or:

```python
N_train = len(train_dataset)          # if using TensorDataset directly
```

The number of training samples is used to calculate the value of `mu`. For proper implementation, it is preferred to tune the `mu_factor` using Optuna. Some datasets may require a `mu_factor` as low as 0.07 or lower. Therefore, Optuna is preferred for finding the best `mu_factor`.

If Optuna is not available or is computationally expensive, a grid search can be used to find the best `mu_factor`, with validation metrics being monitored.

The default `mu_factor` in the code is `1.0`, which is the default argument in:

```python
compute_mu(N_train, mu_factor=1.0)
```

This value corresponds to the base scaling recommended by the paper:

```text
mu = 1.82 * N_train^(-0.55)
```

Users can easily override it by passing a different `mu_factor` (e.g., from Optuna tuning) when calling `compute_mu()`.

**Important:** `mu` itself is not hardcoded anywhere in the model. It must be computed during training and applied to the loss. The helper function simply provides the recommended formula.

Users should call it before training, for example:

```python
mu = compute_mu(N_train, mu_factor=1.0)   # or any tuned value
```

`mu_factor=1.0` is a sensible starting point for most datasets.

### LDIFSequential

LDIFSequential is designed for time-series data (e.g., sensor readings, financial sequences). Its defaults reflect the need for stable recurrent dynamics and awareness of temporal order:

* `num_layers=2`: Same as static — two layers are sufficient for most sequence lengths up to ~100 time steps.
* `k_max=16`: The spectrum capacity is kept at 16, which is enough for typical sequential patterns; the L1 penalty will still prune irrelevant components.
* `r=8`: Gate projection rank remains 8, offering sufficient flexibility for mixing symmetric and skew-symmetric responses.
* `gamma=0.2`: A stronger decay coefficient is used to dissipate energy more quickly, preventing the hidden state from becoming unstable over long sequences.
* `input_scale=0.05`: Input forcing is much weaker compared to the static variant. This ensures that new observations gently influence the state without overwhelming the recurrent dynamics.
* `z_init_mean=-2.0`: The spectrum logits start with a mean of -2.0, corresponding to sigmoid values around 0.12. This highly conservative initialization avoids sudden large updates early in training, which is crucial for stable recurrent learning.
* `z_init_std=0.01`: As with the static variant, the tight standard deviation keeps initial spectrum values close to each other, reducing variance in early gradients.
* `use_positional_encoding=True`: Positional encoding is enabled to inject temporal order information, helping the model distinguish between different time steps and better capture sequential patterns.

`mu` is calculated as described above.

## Hyperparameter Tuning

The defaults work well for moderate-sized datasets (up to ~10k samples for static data and sequences up to length ~100). For larger or more complex data, we recommend tuning the following with Optuna or a similar optimization framework:

* `k_max`: 12–64 (larger for high-capacity tasks)
* `r`: 6–24 (larger for richer gate interactions)
* `num_layers`: 1–4 (more layers for long-range dependencies)
* `gamma`: 0.001–0.1 (static) or 0.05–0.5 (sequential)
* `lr` (optimizer): typically 1e-4 to 1e-3
* `weight_decay`: 1e-5 to 1e-2
* `mu` (spectrum penalty coefficient): use `mu = 1.82 * (N_train ** (-0.55)) * mu_factor`, where `mu_factor` is tuned (e.g., 0.01–10).

A good starting point for the L1 penalty is `mu = 0.01` for datasets with >10k samples, scaling down for smaller datasets.

The initial draft of the paper explaining the model is titled **"LDIF: Latent Dual Interaction Flow"** and is currently under process and will be published on arXiv. For more details, please refer to the paper.

## Difference Between LDIFStatic and LDIFSequential

The underlying model architecture is the same. The main difference is that `LDIFStatic` is designed for static/tabular data. It uses stronger input scaling (`2.0`) and lower decay (`gamma=0.01`) to capture feature interactions, with positional encoding disabled.

`LDIFSequential` targets time-series data. It uses weaker input scaling (`0.05`), stronger decay (`gamma=0.2`) for stable recurrent dynamics, and enables positional encoding to preserve temporal order.

For the `z_init_mean` value, or the initialization of the spectrum values, the static variant uses a value of `1.0`, so the spectrum starts near `0.73`. For the sequential variant, to promote stability and prevent any type of exploding gradients, the value is set to `-2.0`, so the spectrum value starts near `0.12`.

In our experiments, these configurations for the initialization and input scaling were found to work effectively for their respective data types.
