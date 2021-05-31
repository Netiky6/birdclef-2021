import os
from pathlib import Path
import torch
from warnings import simplefilter
import numpy as np


simplefilter(action="ignore", category=UserWarning)
simplefilter(action="ignore", category=FutureWarning)

SEED = 2020

DATA_PATH = Path("/root/kaggle/kaggle_birdcall_identification/input/birdclef-2021")
AUDIO_PATH = DATA_PATH / "train_short_audio"
BACKGROUND_PATH = Path('/root/kaggle/kaggle_birdcall_identification/input/bird-backgrounds')

OUTPUT_PATH = Path("/root/kaggle/kaggle_birdcall_identification/build/")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

NUM_WORKERS = 12

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NUM_CLASSES = 397

CLASSES = sorted(os.listdir(AUDIO_PATH))
resnest50_path = '/data/kaggle/input/resnest50/resnest50-528c19ca.pth'