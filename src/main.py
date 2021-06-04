import argparse
import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from util import seed_everything, count_parameters, save_model_weights
from params import NUM_CLASSES, DATA_PATH, AUDIO_PATH, OUTPUT_PATH
from logger import create_logger
from data.dataset import BirdDataset
from model_zoo.models import get_model
from training.train import fit


class AudioParams:
    """
    Parameters used for the audio data
    """
    sr = 32000
    duration = 5

    # Melspectrogram
    n_mels = 128
    fmin = 20
    fmax = 16000


def train(config, df_train, df_val, fold, cp_folder):
    """
    Trains and validate a model

    Arguments:
        config {Config} -- Parameters
        df_train {pandas dataframe} -- Training metadata
        df_val {pandas dataframe} -- Validation metadata
        fold {int} -- Selected fold

    Returns:
        np array -- Validation predictions
    """

    print(f"    -> {len(df_train)} training birds")
    print(f"    -> {len(df_val)} validation birds")

    seed_everything(config.seed)

    model = get_model(config.selected_model, num_classes=NUM_CLASSES).cuda()
    model.zero_grad()

    train_dataset = BirdDataset(df_train, AudioParams, use_conf=config.use_conf)
    val_dataset = BirdDataset(df_val, AudioParams, train=False)

    n_parameters = count_parameters(model)
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit(
        model,
        train_dataset,
        val_dataset,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        alpha=config.alpha,
        mixup_proba=config.mixup_proba,
        verbose_eval=config.verbose_eval,
        epochs_eval_min=config.epochs_eval_min,
    )

    if config.save:
        save_model_weights(
            model,
            f"{config.selected_model}_{config.name}_{fold}.pt",
            cp_folder=cp_folder,
        )

    return pred_val


def k_fold(config, df, fold):
    """
    Performs a k-fold cross validation

    Arguments:
        config {Config} -- Parameters
        df {pandas dataframe} -- Metadata

    Keyword Arguments:
        df_extra {pandas dataframe or None} -- Metadata of the extra samples to use (default: {None})

    Returns:
        np array -- Out-of-fold predictions
    """

    skf = StratifiedKFold(n_splits=config.k, shuffle=True, random_state=config.random_state)
    splits = list(skf.split(X=df, y=df["primary_label"]))

    pred_oof = np.zeros((len(df), NUM_CLASSES))

    # Checkpoints folder
    TODAY = str(datetime.date.today())

    for i, (train_idx, val_idx) in enumerate(splits):
        if i == fold:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()

            CP_TODAY = OUTPUT_PATH / 'checkpoints' / TODAY / str(i)
            CP_TODAY.mkdir(parents=True, exist_ok=True)

            pred_val = train(config, df_train, df_val, i, CP_TODAY)
            pred_oof[val_idx] = pred_val

    return pred_oof


class Config:
    """
    Parameter used for training
    """
    # General
    seed = 42
    verbose = 1
    verbose_eval = 1
    epochs_eval_min = 25
    save = True

    # k-fold
    k = 5
    random_state = 42

    # Model
    selected_model = "resnest50_fast_1s1x64d"
    # selected_model = "resnext101_32x8d_wsl"
    # selected_model = 'resnext50_32x4d'

    use_conf = False

    # Training
    batch_size = 64
    epochs = 40
    lr = 1e-3
    warmup_prop = 0.05
    val_bs = 64

    if "101" in selected_model or "b5" in selected_model or "b6" in selected_model:
        batch_size = batch_size // 2
        lr = lr / 2

    mixup_proba = 0.5
    alpha = 5

    name = "double"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', '-f', required=True, type=int)
    args = parser.parse_args()

    # Data
    df_train = pd.read_csv(DATA_PATH / "train_metadata.csv")

    df_train["file_path"] = [str(AUDIO_PATH / primary_label / filename) for primary_label, filename
                             in zip(df_train.primary_label, df_train.filename)]

    # Logger
    TODAY = str(datetime.date.today())
    create_logger(str(OUTPUT_PATH), f"{TODAY}_{Config.selected_model}_{Config.name}")

    # Training
    pred_oof = k_fold(Config, df_train, args.fold)
