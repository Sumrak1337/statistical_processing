import os

from pathlib import Path

PROJECT_PATH = Path(__file__).parent.absolute()
CONFIGS_ROOT = PROJECT_PATH / 'configs'
RESULTS_ROOT = PROJECT_PATH / 'results'
DATA_ROOT = PROJECT_PATH / 'data'
CLEAR_DATA_ROOT = DATA_ROOT / 'clear_data_root'

os.makedirs(CLEAR_DATA_ROOT, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)
