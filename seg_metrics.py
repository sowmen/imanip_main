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
        if(c[0].requires_grad or c[1].requires_grad):
            print("NOT DETACHED dice_coeff")

        d = losses.functional.soft_dice_score(c[0], c[1], smooth=1e-7)
        if torch.sum(c[1]).item() > 0 and d.item() > mx_dice:
            mx_dice = d.item()
            best_idx = i 
        s = s + d
        
    print(f"Best Dice: {mx_dice}, Count = {torch.sum(targets[best_idx]).item()}, IDX : {best_idx}")
    return s / (i + 1), (best_idx, mx_dice)

def get_avg_batch_dice(outputs : torch.tensor, targets : torch.tensor): 
    """Dice coeff for batches"""

    tot_dice = 0.0
    mx_dice, worst_dice = -1, 1e9
    best_idx, worst_idx = 0, 0

    for i, c in enumerate(zip(outputs, targets)):

        assert c[0].shape == c[1].shape
        assert c[0].requires_grad == False
        
        d = losses.functional.soft_dice_score(c[0], c[1], smooth=1e-7).item()
        tot_dice += d
        if (d > 0 and d < 1.0) and d > mx_dice:
            mx_dice = d
            best_idx = i
        if (d > 0 and d < 1.0) and d < worst_dice:
            worst_dice = d
            worst_idx = i 
    
    return tot_dice / (outputs.shape[0]), (mx_dice, best_idx), (worst_dice, worst_idx)

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
        if(c[0].requires_grad or c[1].requires_grad):
            print("NOT DETACHED jaccard_coeff")

        d = losses.functional.soft_jaccard_score(c[0], c[1], smooth=1e-7)
        if torch.sum(c[1]).item() > 0 and d.item() > mx_iou:
            mx_iou = d.item()
            best_idx = i 
        s = s + d
        
    print(f"Best IOU: {mx_iou}, Count = {torch.sum(targets[best_idx]).item()}, IDX : {best_idx}")
    return s / (i + 1), (best_idx, mx_iou)

def get_avg_batch_jaccard(outputs : torch.tensor, targets : torch.tensor): 
    """Dice coeff for batches"""

    tot_iou = 0.0
    mx_iou, worst_iou = -1, 1e9
    best_idx, worst_idx = 0, 0

    for i, c in enumerate(zip(outputs, targets)):

        assert c[0].shape == c[1].shape
        assert c[0].requires_grad == False
        
        d = losses.functional.soft_jaccard_score(c[0], c[1], smooth=1e-7).item()
        tot_iou += d
        if (d > 0 and d < 1.0) and d > mx_iou:
            mx_iou = d
            best_idx = i 
        if (d > 0 and d < 1.0) and d < worst_iou:
            worst_iou = d
            worst_idx = i 
    
    return tot_iou / (outputs.shape[0]), (mx_iou, best_idx), (worst_iou, worst_idx)

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


#region DICE TEST
####################################################################################
def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)
        
        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])
        
        # dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        # dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        # dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)
    return dice, dice_neg, dice_pos, num_neg, num_pos

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou

class DiceMeter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        # probs = torch.sigmoid(outputs)
        probs = outputs
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice)
        self.dice_pos_scores.extend(dice_pos)
        self.dice_neg_scores.extend(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou
    
def epoch_score_log(phase, meter):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("%s -> dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f" % (phase, dice, dice_neg, dice_pos, iou))
    return dice, dice_neg, dice_pos, iou
#######################################################################################
#endregion