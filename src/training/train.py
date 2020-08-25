import gc
import time
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from transformers import get_linear_schedule_with_warmup

from util import f1
from params import NUM_WORKERS, NUM_CLASSES
from training.mixup import mixup_data, mixup_criterion


def fit(
    model,
    train_dataset,
    val_dataset,
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    alpha=0.4,
    mixup_proba=0.,
    verbose=1,
    verbose_eval=1,
):
    """
    Usual torch fit function
    
    Arguments:
        model {torch model} -- Model to train
        train_dataset {torch dataset} -- Dataset to train with
        val_dataset {torch dataset} -- Dataset to validate with
        class_weights {numpy array} -- Class weighting in the CE loss to handle inbalance
    
    Keyword Arguments:
        epochs {int} -- Number of epochs (default: {50})
        batch_size {int} -- Training batch size (default: {32})
        batch_size {int} -- Validation batch size (default: {32})
        warmup_prop {float} -- Warmup proportion (default: {0.1})
        lr {[float]} -- Start (or maximum) learning rate (default: {1e-3})
        verbose {[int]} -- Period (in epochs) to display logs at (default: {1})
        verbose {[int]} -- Period (in epochs) to perform evaluation at (default: {1})

    Returns:
        numpy array -- Predictions at the last epoch
    """
    avg_val_loss = 0
    avg_loss = 0
    score = 0

    optimizer = Adam(model.parameters(), lr=lr)

    loss_fct = nn.BCEWithLogitsLoss(reduction="mean").cuda()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False, num_workers=NUM_WORKERS
    )

    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        optimizer.zero_grad()

        avg_loss = 0
        for step, (x, y_batch) in enumerate(train_loader):
            
            if np.random.rand() < mixup_proba:
                x, y_a, y_b ,_ = mixup_data(x.cuda(), y_batch.cuda(), alpha=alpha)
                y_batch = torch.clamp(y_a + y_b, 0, 1)  # I don't use the mixup criterion to help the model robustness

            y_pred = model(x.cuda())
            loss = loss_fct(y_pred, y_batch.cuda().float())

            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if (epoch + 1) % verbose_eval == 0 or (epoch + 1 == epochs):
            model.eval()

            avg_val_loss = 0.0
            with torch.no_grad():
                preds = np.empty((0, NUM_CLASSES))
                for x, y_batch in val_loader:
                    y_pred = model(x.cuda()).detach()
                    loss = loss_fct(y_pred, y_batch.cuda().float())
                    avg_val_loss += loss.item() / len(val_loader)

                    preds = np.concatenate([preds, torch.sigmoid(y_pred).cpu().numpy()])

            score = f1(val_dataset.y, preds)

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_lr()[0]
            print(
                f"Epoch {epoch + 1}/{epochs} \t lr={lr:.1e} \t t={elapsed_time:.0f}s  \t loss={avg_loss:.3f} \t ",
                end="",
            )
            if (epoch + 1) % verbose_eval == 0 or (epoch + 1 == epochs):
                print(f"val_loss={avg_val_loss:.3f} \t val_f1={score:.3f}")
            else:
                print('')

    torch.cuda.empty_cache()
    return preds



def predict(model, dataset, batch_size=64, tta=False):
    """
    Usual torch predict function

    Arguments:
        model {torch model} -- Model to predict with
        dataset {torch dataset} -- Dataset to predict with on

    Keyword Arguments:
        batch_size {int} -- Batch size (default: {32})
        tta {bool} -- Whether to use 4 flips tta (default: {False})

    Returns:
        numpy array -- Predictions
    """
    model.eval()
    preds = np.empty((0))
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    with torch.no_grad():
        for x, y_batch in tqdm(loader):
            y_pred = model(x.cuda()).detach().view(-1)
            probas = torch.sigmoid(y_pred).cpu().numpy()

            if tta:
                flips = [[-1], [-2], [-2, -1]]
                for f in flips:
                    y_pred = model(torch.flip(x.cuda(), f)).view(-1)
                    probas += torch.sigmoid(y_pred).cpu().numpy()
                
                probas /= len(flips) + 1
           
            preds = np.concatenate([preds, probas])

    return preds