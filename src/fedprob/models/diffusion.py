"""Model diffusion (DDPM) bersyarat untuk membangkitkan *jalur* masa depan.

Model kuantil hanya memberi distribusi tiap hari secara terpisah (marginal). Ia tidak
bisa menjawab pertanyaan seperti "berapa peluang saldo turun di bawah X **kapan pun**
dalam 14 hari ke depan?", karena itu butuh distribusi *bersama* seluruh jalur.

Model diffusion belajar membangkitkan jalur lengkap:
- **Training:** ambil jalur masa depan asli x0, tambahkan noise Gaussian sebanyak k
  langkah (x_k), lalu latih jaringan menebak noise tersebut, dengan syarat histori
  (konteks) dan fitur kalender.
- **Sampling:** mulai dari noise murni, lalu "bersihkan" langkah demi langkah sampai
  menjadi jalur yang realistis. Ulangi ratusan kali -> ratusan skenario masa depan.

Arsitektur sengaja kecil (MLP) agar bisa dilatih di CPU dan juga secara federated.
Referensi: Ho dkk. 2020 (DDPM); Rasul dkk. 2021 (TimeGrad); Tashiro dkk. 2021 (CSDI).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import QUANTILES
from ..data.windows import HORIZON, LOOKBACK, N_CAL


def _cosine_alpha_bar(steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, steps, steps + 1) / steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return f / f[0]


class _StepEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=k.device) / half)
        ang = k.float()[:, None] * freqs[None, :]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=1))


class DiffusionForecaster(nn.Module):
    def __init__(
        self,
        lookback: int = LOOKBACK,
        horizon: int = HORIZON,
        n_cal: int = N_CAL,
        steps: int = 50,
        hidden: int = 256,
        ctx_dim: int = 128,
        quantiles=QUANTILES,
        n_samples: int = 200,
        y_scale: float = 0.6,  # ~std selisih target terhadap nilai terakhir (skala ternormalisasi)
        clip: float = 6.0,  # batas |x0| saat sampling, dalam satuan y_scale
        eval_samples: int = 32,  # sampel untuk loss validasi (0 = pakai loss tebak-noise)
    ):
        super().__init__()
        self.horizon, self.steps, self.n_samples, self.y_scale = horizon, steps, n_samples, y_scale
        self.clip, self.eval_samples = clip, eval_samples
        self.quantiles = tuple(quantiles)
        self.register_buffer("q", torch.tensor(self.quantiles))

        ab = _cosine_alpha_bar(steps)
        betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, 0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("post_var", betas * (1 - alpha_bar_prev) / (1 - alpha_bar))

        self.ctx = nn.Sequential(
            nn.Linear(lookback + horizon * n_cal, hidden), nn.SiLU(), nn.Linear(hidden, ctx_dim)
        )
        self.temb = _StepEmbedding(64)
        self.den = nn.Sequential(
            nn.Linear(horizon + 64 + ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, horizon),
        )

    # ------------------------------------------------------------------ inti
    def _context(self, x_hist, x_cal):
        last = x_hist[:, -1:]
        return self.ctx(torch.cat([x_hist - last, x_cal.flatten(1)], dim=1)), last

    def _eps(self, x_k, k, ctx):
        return self.den(torch.cat([x_k, self.temb(k), ctx], dim=1))

    def _loss(self, x_hist, x_cal, y, generator=None):
        ctx, last = self._context(x_hist, x_cal)
        x0 = (y - last) / self.y_scale
        k = torch.randint(0, self.steps, (len(y),), generator=generator)
        noise = torch.randn(x0.shape, generator=generator)
        ab = self.alpha_bar[k][:, None]
        x_k = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return F.mse_loss(self._eps(x_k, k, ctx), noise)

    # ------------------------------------------- antarmuka loop training/federated
    def training_loss(self, x_hist, x_cal, y):
        return self._loss(x_hist, x_cal, y)

    @torch.no_grad()
    def eval_loss(self, x_hist, x_cal, y):
        """Untuk memilih epoch/ronde terbaik. Loss tebak-noise ternyata tidak selaras
        dengan kualitas forecast, jadi dipakai pinball loss dari sampel (seed tetap)."""
        if self.eval_samples <= 0:
            return self._loss(x_hist, x_cal, y, torch.Generator().manual_seed(0))
        # cukup sepertiga origin validasi (origin harian saling tumpang tindih) -> 3x lebih cepat
        x_hist, x_cal, y = x_hist[::3], x_cal[::3], y[::3]
        paths = self.sample(x_hist, x_cal, n_samples=self.eval_samples, seed=0)
        q = torch.quantile(paths, self.q, dim=1).permute(1, 2, 0)
        diff = y.unsqueeze(-1) - q
        return torch.maximum(self.q * diff, (self.q - 1) * diff).mean()

    @torch.no_grad()
    def sample(self, x_hist, x_cal, n_samples: int | None = None, seed: int = 0) -> torch.Tensor:
        """Bangkitkan `n_samples` jalur per input. Return (B, n_samples, H), skala ternormalisasi."""
        n = n_samples or self.n_samples
        g = torch.Generator().manual_seed(seed)
        ctx, last = self._context(x_hist, x_cal)
        B = len(x_hist)
        ctx = ctx.repeat_interleave(n, 0)
        x = torch.randn((B * n, self.horizon), generator=g)
        for k in reversed(range(self.steps)):
            kk = torch.full((B * n,), k, dtype=torch.long)
            eps = self._eps(x, kk, ctx)
            a, ab, b = self.alphas[k], self.alpha_bar[k], self.betas[k]
            ab_prev = self.alpha_bar[k - 1] if k > 0 else torch.tensor(1.0)
            # Tebak jalur bersih x0, batasi ke rentang wajar (mencegah akumulasi galat), lalu
            # hitung rata-rata posterior q(x_{k-1} | x_k, x0).
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-self.clip, self.clip)
            x = (b * ab_prev.sqrt() / (1 - ab)) * x0 + ((1 - ab_prev) * a.sqrt() / (1 - ab)) * x
            if k > 0:
                x = x + self.post_var[k].sqrt() * torch.randn(x.shape, generator=g)
        return (x.view(B, n, self.horizon) * self.y_scale) + last[:, None, :]

    @torch.no_grad()
    def predict_quantiles(self, x_hist, x_cal):
        paths = self.sample(x_hist, x_cal)
        return torch.quantile(paths, self.q, dim=1).permute(1, 2, 0)
