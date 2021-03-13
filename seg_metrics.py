import torch
from pytorch_toolbelt import losses
import numpy as np

def dice_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
):

    outputs = torch.sigmoid(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = ((2 * intersection) + eps) / (union + eps)
    
    return dice #single float

def dice_coeff(outputs : list, targets : list): 
    """Dice coeff for batches"""
    # if outputs.is_cuda:
    #     s = torch.FloatTensor(1).cuda().zero_()
    # else:
    assert len(outputs) == len(targets)
    # assert outputs[0].size() == targets[0].size()

    s = torch.FloatTensor(1).zero_()
    mx_dice = -1
    best_idx = 0

    for i, c in enumerate(zip(outputs, targets)):
        # print(c[0].shape, c[1].shape)
        d = losses.functional.soft_dice_score(c[0], c[1], smooth=1e-7)
        if torch.sum(c[1]).item() > 0 and d.item() > mx_dice:
            mx_dice = d.item()
            best_idx = i 
        s = s + d
        
    print(f"Best Dice: {mx_dice}, Count = {torch.sum(targets[best_idx]).item()}, IDX : {best_idx}")
    return s / (i + 1), (best_idx, mx_dice)

def jaccard_coeff(outputs, targets):
    """jaccard coeff for batches"""
    # if outputs.is_cuda:
    #     s = torch.FloatTensor(1).cuda().zero_()
    # else:
    assert len(outputs) == len(targets)
    # assert outputs[0].size() == targets[0].size()

    s = torch.FloatTensor(1).zero_()
    mx_iou = -1
    best_idx = 0
    
    for i, c in enumerate(zip(outputs, targets)):
        d = losses.functional.soft_jaccard_score(c[0], c[1], smooth=1e-7)
        if torch.sum(c[1]).item() > 0 and d.item() > mx_iou:
            mx_iou = d.item()
            best_idx = i 
        s = s + d
        
    print(f"Best IOU: {mx_iou}, Count = {torch.sum(targets[best_idx]).item()}, IDX : {best_idx}")
    return s / (i + 1), (best_idx, mx_iou)

def sensitivity(outputs, targets):
    true_positives = torch.sum(torch.round(torch.clamp(targets * outputs, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(targets, 0, 1)))
    return true_positives / (possible_positives + 1e-7)

def specificity(outputs, targets):
    true_negatives = torch.sum(torch.round(torch.clamp((1 - targets) * (1 - outputs), 0, 1)))
    possible_negatives = torch.sum(torch.round(torch.clamp(1 - targets, 0, 1)))
    return true_negatives / (possible_negatives + 1e-7)

# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow
def get_iou_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric