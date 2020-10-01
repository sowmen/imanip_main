import torch
from pytorch_toolbelt import losses

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

def dice_coeff_single(outputs, targets):
    """Dice coeff for batches"""
    if outputs.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(outputs, targets)):
        s = s + losses.functional.soft_dice_score(c[0], c[1])

    return s / (i + 1)

def jaccard_coeff_single(outputs, targets):
    """jaccard coeff for batches"""
    if outputs.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(outputs, targets)):
        s = s + losses.functional.soft_jaccard_score(c[0], c[1])

    return s / (i + 1)

def sensitivity(outputs, targets):
    true_positives = torch.sum(torch.round(torch.clamp(targets * outputs, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(targets, 0, 1)))
    return true_positives / (possible_positives + 1e-7)

def specificity(outputs, targets):
    true_negatives = torch.sum(torch.round(torch.clamp((1 - targets) * (1 - outputs), 0, 1)))
    possible_negatives = torch.sum(torch.round(torch.clamp(1 - targets, 0, 1)))
    return true_negatives / (possible_negatives + 1e-7)