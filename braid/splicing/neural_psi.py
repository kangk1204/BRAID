"""Neural posterior estimation for PSI quantification.

Replaces the analytic Beta-binomial model with a mixture density network
(MDN) that learns the posterior distribution of PSI values conditioned on
junction read counts and event features. This captures complex, multi-modal
posteriors that arise from biological variability across replicates.

Falls back to the Beta-binomial model when PyTorch is unavailable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception as exc:
    logger.debug("PyTorch unavailable for neural PSI fallback: %s", exc)

# Input feature dimension for the MDN
PSI_FEATURE_DIM = 8
# Number of mixture components
N_COMPONENTS = 3


@dataclass
class NeuralPSIResult:
    """Result from neural PSI estimation.

    Attributes:
        psi_mean: Posterior mean PSI estimate.
        psi_mode: Posterior mode (MAP) estimate.
        ci_low: Lower bound of 95% credible interval.
        ci_high: Upper bound of 95% credible interval.
        mixture_weights: Mixture component weights.
        mixture_means: Mixture component means.
        mixture_stds: Mixture component standard deviations.
    """

    psi_mean: float
    psi_mode: float
    ci_low: float
    ci_high: float
    mixture_weights: list[float]
    mixture_means: list[float]
    mixture_stds: list[float]


def extract_psi_features(
    inclusion_count: int,
    exclusion_count: int,
    n_inclusion_junctions: int,
    n_exclusion_junctions: int,
    event_type: int,
) -> np.ndarray:
    """Extract feature vector for neural PSI estimation.

    Args:
        inclusion_count: Number of inclusion junction reads.
        exclusion_count: Number of exclusion junction reads.
        n_inclusion_junctions: Number of inclusion junctions.
        n_exclusion_junctions: Number of exclusion junctions.
        event_type: Integer event type (0-6).

    Returns:
        Feature vector of shape (PSI_FEATURE_DIM,).
    """
    total = inclusion_count + exclusion_count
    feats = np.zeros(PSI_FEATURE_DIM, dtype=np.float32)

    feats[0] = math.log1p(inclusion_count)
    feats[1] = math.log1p(exclusion_count)
    feats[2] = math.log1p(total)
    feats[3] = inclusion_count / max(total, 1)
    feats[4] = n_inclusion_junctions
    feats[5] = n_exclusion_junctions
    feats[6] = event_type / 6.0  # Normalized event type
    feats[7] = 1.0 / max(math.sqrt(total), 1.0)  # Inverse sqrt for uncertainty

    return feats


def beta_binomial_fallback(
    inclusion_count: int,
    exclusion_count: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> NeuralPSIResult:
    """Compute PSI using the Beta-binomial posterior as fallback.

    Args:
        inclusion_count: Number of inclusion reads.
        exclusion_count: Number of exclusion reads.
        alpha_prior: Beta distribution alpha prior.
        beta_prior: Beta distribution beta prior.

    Returns:
        NeuralPSIResult with Beta-binomial estimates.
    """
    from scipy.stats import beta

    alpha = alpha_prior + inclusion_count
    beta_param = beta_prior + exclusion_count

    mean = alpha / (alpha + beta_param)
    if alpha > 1 and beta_param > 1:
        mode = (alpha - 1) / max(alpha + beta_param - 2, 1e-10)
    else:
        mode = mean
    mode = max(0.0, min(1.0, mode))

    ci_low = float(beta.ppf(0.025, alpha, beta_param))
    ci_high = float(beta.ppf(0.975, alpha, beta_param))

    return NeuralPSIResult(
        psi_mean=mean,
        psi_mode=mode,
        ci_low=ci_low,
        ci_high=ci_high,
        mixture_weights=[1.0],
        mixture_means=[mean],
        mixture_stds=[
            math.sqrt(
                alpha * beta_param
                / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
            )
        ],
    )


if _TORCH_AVAILABLE:

    class MixtureDensityNetwork(nn.Module):
        """Mixture Density Network for PSI posterior estimation.

        Architecture: Input -> Linear(8->32) -> ReLU -> Linear(32->16) ->
        ReLU -> outputs (weights, means, log_vars) for K mixture components.

        The output parametrizes a mixture of K Beta-like distributions
        (logit-normal components) over [0, 1].
        """

        def __init__(
            self,
            input_dim: int = PSI_FEATURE_DIM,
            hidden_dim: int = 32,
            n_components: int = N_COMPONENTS,
        ) -> None:
            super().__init__()
            self.n_components = n_components
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            out_dim = hidden_dim // 2
            self.fc_weights = nn.Linear(out_dim, n_components)
            self.fc_means = nn.Linear(out_dim, n_components)
            self.fc_log_vars = nn.Linear(out_dim, n_components)

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                x: Input features (batch, input_dim).

            Returns:
                Tuple of (weights, means, log_vars), each (batch, n_components).
                weights: Softmax mixture weights.
                means: Sigmoid-clamped means in [0, 1].
                log_vars: Log-variance for each component.
            """
            h = self.encoder(x)
            weights = torch.softmax(self.fc_weights(h), dim=-1)
            means = torch.sigmoid(self.fc_means(h))
            log_vars = self.fc_log_vars(h).clamp(-6, 2)
            return weights, means, log_vars

    def _mdn_loss(
        weights: torch.Tensor,
        means: torch.Tensor,
        log_vars: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss for the MDN.

        Args:
            weights: Mixture weights (batch, K).
            means: Component means (batch, K).
            log_vars: Component log-variances (batch, K).
            targets: Target PSI values (batch,).

        Returns:
            Scalar loss.
        """
        targets = targets.unsqueeze(-1)  # (batch, 1)
        vars_ = torch.exp(log_vars)  # (batch, K)
        log_probs = (
            -0.5 * ((targets - means) ** 2) / vars_
            - 0.5 * log_vars
            - 0.5 * math.log(2 * math.pi)
        )
        weighted = torch.log(weights + 1e-10) + log_probs
        return -torch.logsumexp(weighted, dim=-1).mean()


class NeuralPSIEstimator:
    """Neural posterior PSI estimator with MDN.

    Uses a Mixture Density Network to estimate the full posterior
    distribution of PSI, providing richer uncertainty quantification
    than the analytic Beta-binomial model.

    Args:
        model_path: Optional path to saved MDN weights.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model: object | None = None
        self._is_trained = False

        if model_path is not None and _TORCH_AVAILABLE:
            try:
                self._model = MixtureDensityNetwork()
                state = torch.load(
                    model_path, map_location="cpu", weights_only=True
                )
                self._model.load_state_dict(state)
                self._model.eval()
                self._is_trained = True
                logger.info("Loaded neural PSI estimator from %s", model_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load neural PSI estimator: %s", exc
                )
                self._model = None

    @property
    def is_trained(self) -> bool:
        """Whether a trained MDN model is loaded."""
        return self._is_trained

    def estimate(
        self,
        inclusion_count: int,
        exclusion_count: int,
        n_inclusion_junctions: int = 1,
        n_exclusion_junctions: int = 1,
        event_type: int = 0,
    ) -> NeuralPSIResult:
        """Estimate PSI posterior for a single event.

        Args:
            inclusion_count: Inclusion junction reads.
            exclusion_count: Exclusion junction reads.
            n_inclusion_junctions: Number of inclusion junctions.
            n_exclusion_junctions: Number of exclusion junctions.
            event_type: Integer event type.

        Returns:
            NeuralPSIResult with posterior estimates.
        """
        if self._is_trained and _TORCH_AVAILABLE and self._model is not None:
            features = extract_psi_features(
                inclusion_count,
                exclusion_count,
                n_inclusion_junctions,
                n_exclusion_junctions,
                event_type,
            )
            with torch.no_grad():
                x = torch.from_numpy(features).float().unsqueeze(0)
                weights, means, log_vars = self._model(x)

            w = weights[0].numpy()
            m = means[0].numpy()
            s = np.sqrt(np.exp(log_vars[0].numpy()))

            # Posterior mean
            psi_mean = float(np.sum(w * m))

            # Posterior mode: component with highest weight
            mode_idx = int(np.argmax(w))
            psi_mode = float(m[mode_idx])

            # Approximate CI by sampling
            n_samples = 1000
            samples = []
            for k in range(len(w)):
                n_k = max(1, int(w[k] * n_samples))
                comp_samples = np.random.normal(m[k], s[k], size=n_k)
                samples.extend(comp_samples.tolist())
            samples_arr = np.clip(samples, 0, 1)
            ci_low = float(np.percentile(samples_arr, 2.5))
            ci_high = float(np.percentile(samples_arr, 97.5))

            return NeuralPSIResult(
                psi_mean=psi_mean,
                psi_mode=psi_mode,
                ci_low=ci_low,
                ci_high=ci_high,
                mixture_weights=w.tolist(),
                mixture_means=m.tolist(),
                mixture_stds=s.tolist(),
            )

        # Fallback to Beta-binomial
        return beta_binomial_fallback(inclusion_count, exclusion_count)

    def train_model(
        self,
        features: np.ndarray,
        psi_targets: np.ndarray,
        n_epochs: int = 200,
        lr: float = 1e-3,
    ) -> float:
        """Train the MDN on observed PSI data.

        Args:
            features: Feature matrix (n_samples, PSI_FEATURE_DIM).
            psi_targets: Target PSI values (n_samples,).
            n_epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Final training loss.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available; cannot train MDN.")
            return float("nan")

        self._model = MixtureDensityNetwork()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        x = torch.from_numpy(features).float()
        y = torch.from_numpy(psi_targets).float()

        self._model.train()
        final_loss = 0.0
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            weights, means, log_vars = self._model(x)
            loss = _mdn_loss(weights, means, log_vars, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self._model.eval()
        self._is_trained = True
        logger.info("Trained neural PSI estimator: final loss=%.4f", final_loss)
        return final_loss

    def save(self, path: str) -> None:
        """Save trained model weights.

        Args:
            path: Output file path.
        """
        if self._is_trained and _TORCH_AVAILABLE and self._model is not None:
            torch.save(self._model.state_dict(), path)
            logger.info("Saved neural PSI estimator to %s", path)
