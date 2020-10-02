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

def dice_coeff(outputs : list, targets : list): 
    """Dice coeff for batches"""
    # if outputs.is_cuda:
    #     s = torch.FloatTensor(1).cuda().zero_()
    # else:
    assert len(outputs) == len(targets)
    assert outputs[0].size() == targets[0].size()

    s = torch.FloatTensor(1).zero_()
    mx_dice = -1
    best_idx = 0

    for i, c in enumerate(zip(outputs, targets)):
        d = losses.functional.soft_dice_score(c[0], c[1])
        if torch.sum(c[1]).item() > 0 and d.item() > mx_dice:
            mx_dice = d.item()
            best_idx = i 
        s = s + d
        
    print(f"Best : {mx_dice}")
    return s / (i + 1)

def jaccard_coeff(outputs, targets):
    """jaccard coeff for batches"""
    # if outputs.is_cuda:
    #     s = torch.FloatTensor(1).cuda().zero_()
    # else:
    assert len(outputs) == len(targets)
    assert outputs[0].size() == targets[0].size()

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