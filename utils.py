import numpy as np
import torch
import torch_optimizer
from torch import optim
import wandb
import cv2
import time    


class EarlyStopping:
    """
    EarlyStopping taken from avishekthakur/wtfml
    """

    def __init__(self, patience=7, mode="max", delta=0.0001, tpu=False):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.tpu = tpu
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )

            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(model, optimizer, learning_rate, weight_decay):
    if optimizer == "radam":
        optimizer = torch_optimizer.RAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "adamP":
        optimizer = torch_optimizer.AdamP(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "qhadam":
        optimizer = torch_optimizer.QHAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    return optimizer


def image2np(image: torch.Tensor) -> np.ndarray:
    "Convert from torch style `image` to numpy/matplotlib style."
    res = image.cpu().permute(1, 2, 0).numpy()
    return res[..., 0] if res.shape[2] == 1 else res                                            


def measure_time(f):

    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('%r, %2.2f sec' % (f.__name__, te-ts))
        return result

    return timed


from io import BytesIO
from PIL import Image, ImageChops

# @measure_time
def get_ela(image, scale):
    
    pil_ori_image = Image.fromarray(image)
    output = BytesIO()
    pil_ori_image.save(output, format="JPEG", quality=90)
    pil_compressed_image = Image.open(output)
    diff = ImageChops.difference(pil_ori_image, pil_compressed_image)
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * scale for k in d[x, y])

    ela = np.array(diff).astype('uint8')

    return ela