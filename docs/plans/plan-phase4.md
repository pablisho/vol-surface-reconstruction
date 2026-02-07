# Phase 4: VAE Baseline — Implementation Plan

## Context

Phase 3 established an E2E training pipeline with MLP, CNN, and U-Net baselines. The VAE is the first model with a latent representation — the established literature baseline (Bergeron et al. 2021, Feugang Nteumagné et al. 2025) against which the transformer (Phase 5) will be measured.

Both reference papers use **fully-connected (dense)** encoder/decoder architectures. We implement the FC version first to match the literature, then add a convolutional variant to see if spatial structure helps the VAE too (as it did for CNN/U-Net vs MLP in Phase 3).

## Architecture

### FC VAE (`VAEReconstructor`) — matches literature

Based on Bergeron et al. (rotmanfinhub/vol-surface-vae-pub): flatten input, FC hidden layers with ReLU, mu/logvar bottleneck, mirrored FC decoder.

```
Encoder:
  (batch, 2, 8, 25) → flatten → (batch, 400)
  → Linear(400, hidden) + ReLU
  → Linear(hidden, hidden) + ReLU
  → fc_mu(hidden, latent_dim), fc_logvar(hidden, latent_dim)

Reparameterization:
  train: z = mu + exp(0.5*logvar) * eps
  eval:  z = mu  (deterministic)

Decoder:
  → Linear(latent_dim, hidden) + ReLU
  → Linear(hidden, hidden) + ReLU
  → Linear(hidden, 200) → reshape → (batch, 1, 8, 25)
```

Default: `hidden_dims=(256, 256)`, `latent_dim=32`, `beta=1.0`.

### Conv VAE (`ConvVAEReconstructor`) — spatial variant

```
Encoder:
  (batch, 2, 8, 25)
  → Conv2d(2, C, 3, stride=1, pad=1) + ReLU       → (batch, C, 8, 25)
  → Conv2d(C, 2C, 3, stride=2, pad=1) + ReLU      → (batch, 2C, 4, 13)
  → Conv2d(2C, 4C, 3, stride=2, pad=1) + ReLU     → (batch, 4C, 2, 7)
  → Flatten → fc_mu(flat, latent_dim), fc_logvar(flat, latent_dim)

Decoder:
  fc_decode(latent_dim, flat) + ReLU → reshape → (batch, 4C, 2, 7)
  → F.interpolate(4, 13) + Conv2d(4C, 2C, 3, pad=1) + ReLU
  → F.interpolate(8, 25) + Conv2d(2C, C, 3, pad=1) + ReLU
  → Conv2d(C, 1, 3, pad=1)                         → (batch, 1, 8, 25)
```

Default: `base_channels=32`, `latent_dim=32`, `beta=1.0`. Decoder uses `F.interpolate` + Conv2d (same pattern as U-Net), avoiding ConvTranspose2d issues.

## Loss: ELBO

`training_loss(pred, target) = MSE(pred, target) + beta * KL`

where `KL = -0.5 * mean(1 + logvar - mu² - exp(logvar))`.

KL computed during `forward()`, stored as `self._kl_loss`. The `training_loss()` method combines it with MSE.

## Training integration

Minimal change to `training/trainer.py` (lines 51 and 73): if the model defines `training_loss(pred, target)`, use it instead of `criterion(pred, target)`. Backward-compatible — existing models don't define it.

```python
# Lines 51, 73: replace `loss = criterion(pred, target)` with:
if hasattr(model, "training_loss"):
    loss = model.training_loss(pred, target)
else:
    loss = criterion(pred, target)
```

## Files

| File | Action | Description |
|------|--------|-------------|
| `models/vae.py` | **Create** | VAEReconstructor (FC) + ConvVAEReconstructor |
| `training/trainer.py` | **Modify** | Add training_loss() dispatch at lines 51, 73 |
| `experiments/train_baseline.py` | **Modify** | Add "vae" and "conv_vae" to build_model() and --model choices |
| `tests/test_models.py` | **Modify** | Add TestVAEReconstructor + TestConvVAEReconstructor (standard tests) |
| `tests/test_vae.py` | **Create** | VAE-specific: KL storage, beta=0, eval determinism, train stochasticity, gradient through KL |

## Implementation order

1. Modify `training/trainer.py` — add training_loss dispatch (4 lines)
2. Create `models/vae.py` — both VAE classes
3. Add both VAEs to `tests/test_models.py` (standard shape/gradient tests)
4. Create `tests/test_vae.py` (VAE-specific tests for both variants)
5. Add trainer dispatch test to `tests/test_trainer.py`
6. Update `experiments/train_baseline.py` — add "vae" and "conv_vae" choices
7. Run full tests + lint + E2E experiment

## Verification

```bash
python -m pytest                                      # all tests pass
python -m ruff check . && python -m ruff format .     # lint clean
python -m experiments.train_baseline --model vae       # FC VAE
python -m experiments.train_baseline --model conv_vae  # Conv VAE
```

Compare RMSE missing: FC VAE vs MLP (~0.010), Conv VAE vs CNN/U-Net (~0.005).

## References

### Primary literature (VAE for volatility surfaces)

- **Bergeron, M., Fung, N., Hull, J., Poulos, Z., & Veneris, A. (2021)**. *Variational Autoencoders: A Hands-Off Approach to Volatility*. arXiv:2102.03945.
  - Paper: https://arxiv.org/abs/2102.03945
  - PDF: https://www-2.rotman.utoronto.ca/~hull/downloadablepublications/Autoencoders_Vol_Sfces.pdf
  - Code: https://github.com/rotmanfinhub/vol-surface-vae-pub
  - **Key influence**: FC encoder/decoder architecture (Dense VAE), reparameterization trick, ELBO loss with beta weighting. Our `VAEReconstructor` follows their `dense_vae.py` pattern: flatten → FC hidden layers with ReLU → mu/logvar → mirrored decoder.

- **Feugang Nteumagné, B., Azemtsa Donfack, H., & Wafo Soh, C. (2025)**. *Variational Autoencoders for Completing the Volatility Surfaces*. Journal of Risk and Financial Management, 18(5), 239.
  - Paper: https://www.mdpi.com/1911-8074/18/5/239
  - Preprint: https://www.preprints.org/manuscript/202502.1482
  - **Key influence**: Direct comparison target — same task (completing incomplete vol surfaces from sparse observations). Demonstrates VAE with latent space optimization outperforms thin-plate spline, SABR, SVI, and deterministic autoencoders.

### Related work (architecture details from related VAE papers)

- **Ning, B., Jaimungal, S., Zhang, X., & Bergeron, M. (2025)**. *Arbitrage-Free Implied Volatility Surface Generation with Variational Autoencoders*. SIAM Journal on Financial Mathematics.
  - Paper: https://arxiv.org/abs/2108.04941
  - Code: https://github.com/BrianNingUT/ArbFreeIV-VAE
  - **Relevance**: Extends Bergeron et al. with arbitrage-free constraints. Provides architecture reference for conditional VAE variants.

- **Ning, B., Jaimungal, S., Zhang, X., & Bergeron, M. (2025)**. *Controllable Generation of Implied Volatility Surfaces with Variational Autoencoders*. arXiv:2509.01743.
  - Paper: https://arxiv.org/abs/2509.01743
  - **Architecture details used**: ResNet encoder/decoder with hidden layers [256, 128] / [128, 256], latent_dim=5, ReLU activation, beta=5e-8. Informed our choice of hidden layer sizes and latent dimension range.

### General VAE theory

- **Kingma, D.P. & Welling, M. (2014)**. *Auto-Encoding Variational Bayes*. arXiv:1312.6114.
  - The foundational VAE paper. ELBO derivation, reparameterization trick, Gaussian encoder/decoder.

- **Higgins, I. et al. (2017)**. *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR 2017.
  - Beta weighting of KL divergence term. Informs our `beta` parameter design.
