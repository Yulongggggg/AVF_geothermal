"""Physical-space adapter for the pinned Equinor ES-MDA 1.1.0 implementation.

Version 1.1.0 stores an upper Cholesky factor ``U`` such that ``R = U.T @ U``.
Its gain calculation correctly uses this convention, but its dense-covariance
noise sampler multiplies ``U @ z``, producing covariance ``U @ U.T`` instead.
Only perturbation generation is overridden here. The original identity-covariance
workflow was unaffected because the two factors then coincide.

This module is imported lazily so the ES-MDA extra remains optional.
"""
import numpy as np
from iterative_ensemble_smoother import ESMDA


class PhysicalESMDA(ESMDA):
    """Reference ES-MDA with perturbations of covariance exactly ``alpha * R``."""

    def perturb_observations(self, *, ensemble_size, alpha):
        """Return ``d[:, None] + sqrt(alpha) * R_cholesky @ z``.

        Keep the library's upper factor intact: its inversion routine needs it.
        Draw the same standard-normal array as the original whitened workflow.
        """
        standard_normal = self.rng.standard_normal(size=(len(self.observations), ensemble_size))
        if self.C_D_L.ndim == 2:
            perturbations = self.C_D_L.T @ standard_normal
        else:
            perturbations = self.C_D_L[:, None] * standard_normal
        return self.observations[:, None] + np.sqrt(alpha) * perturbations.astype(self.observations.dtype)
