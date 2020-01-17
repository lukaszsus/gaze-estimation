import os

DATA_PATH = None
PROJECT_PATH = os.path.dirname(__file__)
RESULTS_PATH = None
MPIIGAZE_PATH = None
UT_MULTIVIEW_PATH = None

try:
    from user_settings import *
except ImportError:
    pass
