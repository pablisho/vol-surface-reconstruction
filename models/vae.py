# models/vae.py
"""VAE models for vol surface reconstruction.

Two variants:
- VAEReconstructor: FC encoder/decoder matching the literature
  (Bergeron et al. 2021, Feugang Nteumagné et al. 2025)
- ConvVAEReconstructor: Convolutional encoder/decoder exploiting
  spatial structure of the strike × maturity grid.

Both use the standard ELBO loss: MSE reconstruction + beta * KL divergence.
KL is computed during forward() and stored for use by training_loss().

References:
    Bergeron et al. (2021) — Variational Autoencoders: A Hands-Off
        Approach to Volatility. arXiv:2102.03945.
    Feugang Nteumagné et al. (2025) — Variational Autoencoders for
        Completing the Volatility Surfaces. JRFM 18(5), 239.
    Kingma & Welling (2014) — Auto-Encoding Variational Bayes.
        arXiv:1312.6114. (ELBO, reparameterization trick)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.base import SurfaceReconstructor


class VAEReconstructor(SurfaceReconstructor):
    """Fully-connected VAE for surface reconstruction (matches literature).

    Input:  (batch, 2, n_taus, n_strikes) — masked IVs + mask channel
    Output: (batch, 1, n_taus, n_strikes) — reconstructed surface

    During forward(), stores self._kl_loss for use by training_loss().
    In eval mode, uses posterior mean (deterministic); in train mode,
    uses reparameterized sample.
    """

    def __init__(
        self,
        n_taus: int,
        n_strikes: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        latent_dim: int = 32,
        beta: float = 1e-4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.n_taus = n_taus
        self.n_strikes = n_strikes
        self.latent_dim = latent_dim
        self.beta = beta

        act_cls: type[nn.Module] = {"relu": nn.ReLU, "elu": nn.ELU}[activation]

        input_dim = 2 * n_taus * n_strikes
        output_dim = n_taus * n_strikes

        # Encoder: input → hidden layers → mu, logvar
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(act_cls())
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder: latent → mirrored hidden layers → output
        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(act_cls())
            prev = h
        dec_layers.append(nn.Linear(prev, output_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self._kl_loss = torch.tensor(0.0)
        self._last_mu = torch.zeros(latent_dim)

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def _decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        flat = x.reshape(batch, -1)

        mu, logvar = self._encode(flat)
        z = self._reparameterize(mu, logvar)
        recon = self._decode(z)

        self._kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        self._last_mu = mu.detach()

        return recon.reshape(batch, 1, self.n_taus, self.n_strikes)

    def training_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """ELBO loss: MSE reconstruction + beta * KL divergence."""
        recon_loss = nn.functional.mse_loss(pred, target)
        return recon_loss + self.beta * self._kl_loss


def latent_optimize(
    model: VAEReconstructor | ConvVAEReconstructor,
    observed_ivs: Tensor,
    mask: Tensor,
    n_steps: int = 200,
    lr: float = 0.01,
) -> Tensor:
    """Reconstruct a surface by optimizing in latent space (Feugang Nteumagné et al. 2025).

    Instead of encoding the partial surface, find the latent z* that minimizes
    MSE between decode(z*) and observed values. This projects the incomplete
    observation onto the learned manifold of complete surfaces.

    Args:
        model: Trained VAE (must be in eval mode, weights frozen).
        observed_ivs: (batch, 1, n_taus, n_strikes) — IV values (any value at missing points).
        mask: (batch, n_taus, n_strikes) — True/1.0 at observed points.
        n_steps: Number of gradient descent steps on z.
        lr: Learning rate for z optimization.

    Returns:
        (batch, 1, n_taus, n_strikes) — reconstructed full surface.
    """
    device = next(model.parameters()).device
    observed_ivs = observed_ivs.to(device)
    mask = mask.to(device).unsqueeze(1)  # (batch, 1, n_taus, n_strikes)

    # Initialize z from encoder via forward pass (works for both FC and Conv VAE)
    with torch.no_grad():
        enc_input = torch.cat([observed_ivs * mask, mask], dim=1)  # (batch, 2, H, W)
        model(enc_input)  # populates model._last_mu
        z = model._last_mu.clone()

    z.requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    out_shape = observed_ivs.shape
    for _ in range(n_steps):
        recon = model._decode(z).reshape(out_shape)
        loss = nn.functional.mse_loss(recon * mask, observed_ivs * mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return model._decode(z).reshape(out_shape)


class ConvVAEReconstructor(SurfaceReconstructor):
    """Convolutional VAE for surface reconstruction (spatial variant).

    Input:  (batch, 2, n_taus, n_strikes) — masked IVs + mask channel
    Output: (batch, 1, n_taus, n_strikes) — reconstructed surface

    Encoder uses stride-2 Conv2d for downsampling; decoder uses
    F.interpolate + Conv2d for upsampling (avoids ConvTranspose2d
    checkerboard artifacts).
    """

    def __init__(
        self,
        n_taus: int,
        n_strikes: int,
        base_channels: int = 32,
        latent_dim: int = 32,
        beta: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_taus = n_taus
        self.n_strikes = n_strikes
        self.latent_dim = latent_dim
        self.beta = beta

        c = base_channels

        # Compute spatial dims after 2 stride-2 downsamples
        h1 = (n_taus + 2 * 1 - 3) // 2 + 1
        w1 = (n_strikes + 2 * 1 - 3) // 2 + 1
        h2 = (h1 + 2 * 1 - 3) // 2 + 1
        w2 = (w1 + 2 * 1 - 3) // 2 + 1
        self._size1 = (h1, w1)
        self._size2 = (h2, w2)
        flat_dim = c * 4 * h2 * w2

        # Encoder
        self.enc_conv1 = nn.Conv2d(2, c, 3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1)
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        self.dec_conv1 = nn.Conv2d(c * 4, c * 2, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(c * 2, c, 3, padding=1)
        self.dec_final = nn.Conv2d(c, 1, 3, padding=1)

        self._kl_loss = torch.tensor(0.0)
        self._last_mu = torch.zeros(latent_dim)

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = torch.relu(self.enc_conv1(x))
        h = torch.relu(self.enc_conv2(h))
        h = torch.relu(self.enc_conv3(h))
        h = h.reshape(h.shape[0], -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def _decode(self, z: Tensor) -> Tensor:
        h = torch.relu(self.fc_decode(z))
        h = h.reshape(h.shape[0], -1, self._size2[0], self._size2[1])
        h = nn.functional.interpolate(h, size=self._size1, mode="bilinear", align_corners=False)
        h = torch.relu(self.dec_conv1(h))
        h = nn.functional.interpolate(
            h, size=(self.n_taus, self.n_strikes), mode="bilinear", align_corners=False
        )
        h = torch.relu(self.dec_conv2(h))
        return self.dec_final(h)

    def forward(self, x: Tensor) -> Tensor:
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        recon = self._decode(z)

        self._kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        self._last_mu = mu.detach()

        return recon

    def training_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """ELBO loss: MSE reconstruction + beta * KL divergence."""
        recon_loss = nn.functional.mse_loss(pred, target)
        return recon_loss + self.beta * self._kl_loss
