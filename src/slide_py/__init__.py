__version__ = "1.0.0"

from .knockoffs import Knockoffs
from .knockoff.filter import VotingResult
from .love import call_love
from .plotting import Plotter
from .score import Estimator, SLIDE_Estimator
from .slide import SLIDE, OptimizeSLIDE
from .cv import SLIDEcv
from .tools import init_data, show_params, check_params, calc_default_fsize
__all__ = [
    '__version__',
    'Knockoffs',
    'VotingResult',
    'call_love',
    'Plotter',
    'Estimator', 'SLIDE_Estimator',
    'SLIDE', 'OptimizeSLIDE',
    'SLIDEcv',
    'init_data', 'show_params', 'check_params', 'calc_default_fsize',
]