"""
LOVE: Latent-model based OVErlapping clustering.

A Python implementation of the LOVE algorithm for overlapping clustering
under structured latent factor models.

Main Functions
--------------
LOVE : Main function for overlapping clustering
Screen_X : Pre-screening to detect pure noise features
KfoldCV_delta : K-fold cross-validation for delta selection

References
----------
Bing, X., Bunea, F., Yang N and Wegkamp, M. (2020)
Adaptive estimation in structured factor models with applications to
overlapping clustering, Annals of Statistics, Vol.48(4) 2055-2081.

Bing, X., Bunea, F. and Wegkamp, M. (2021)
Detecting approximate replicate components of a high-dimensional random
vector with latent structure.
"""

from .love import LOVE
from .prescreen import Screen_X
from .cv import KfoldCV_delta, CV_delta, CV_lbd
from .score import Score_mat
from .est_omega import estOmega
from .est_pure_homo import EstAI, EstC
from .est_pure_hetero import Est_Pure, Est_BI_C
from .est_nonpure import EstY, EstAJInv, EstAJDant
from .utilities import recoverGroup, threshA, offSum

__version__ = "1.0.0"

__all__ = [
    # Main functions
    "LOVE",
    "Screen_X",
    # Cross-validation
    "KfoldCV_delta",
    "CV_delta",
    "CV_lbd",
    # Score computation
    "Score_mat",
    # Estimation functions
    "estOmega",
    "EstAI",
    "EstC",
    "Est_Pure",
    "Est_BI_C",
    "EstY",
    "EstAJInv",
    "EstAJDant",
    # Utilities
    "recoverGroup",
    "threshA",
    "offSum",
]
