import os
from pathlib import Path

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PROJ_ROOT = SRC_ROOT.parent

DATA_ROOT = PROJ_ROOT / "data"
RESULT_PATH = PROJ_ROOT / 'results'

FGC_DEV = DATA_ROOT / "FGC" / "FGC_release_1.7.13" / "FGC_release_all_dev.json"
FGC_TRAIN = DATA_ROOT / "FGC" / "FGC_release_1.7.13" / "FGC_release_all_train.json"
FGC_TEST = DATA_ROOT / "FGC" / "FGC_release_1.7.13" / "FGC_release_all_test.json"

TRAINED_MODELS = RESULT_PATH / "trainedmodels"
