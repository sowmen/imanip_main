import cv2
import torch
import numpy as np
import albumentations
from albumentations import augmentations

def patch_func(img, ela, patch_size):
    d = img.shape
    patches = []
    for i in range(0, d[0], patch_size):
        for j in range(0, d[1], patch_size):
            x = i + patch_size
            y = j + patch_size
            if x > d[0] or y > d[1]:
                break
            temp_img = img[i: x, j: y]
            temp_ela = ela[i: x, j: y]
            # temp = cv2.resize(temp, (d[0],d[1]), interpolation=cv2.INTER_AREA)
            patches.append((temp_img, temp_ela))
    return patches


def ensemble(model, image, ela_image):
    normalize = {
        "mean": [0.42468103282400615, 0.4259826707370029, 0.38855473517307415],
        "std": [0.2744059987371694, 0.2684138285232067, 0.29527622263685294],
    }
    valid_aug = albumentations.Compose(
        [
            augmentations.transforms.Resize(224, 224, interpolation=cv2.INTER_AREA, always_apply=True, p=1),
            albumentations.Normalize(mean=normalize['mean'], std=normalize['std'], always_apply=True, p=1),
            albumentations.pytorch.ToTensor()
        ],
        additional_targets={'ela':'image'}
    )

    model.eval()
    res = []
    # print("=====>1 ", next(model.parameters()).device)
    model.load_state_dict(torch.load('best_weights/CASIA_FULL_ELA.h5'))
    trans = valid_aug(image=image, ela=ela_image)
    image_tensor = trans["image"].unsqueeze(0).cuda()
    ela_tensor = trans["ela"].unsqueeze(0).cuda()
    with torch.no_grad():
        _, (_, enc_out, _, _) = model(image_tensor, ela_tensor)
        res.append(enc_out.cpu().detach())
        # print(enc_out.shape)

    # print("=====>2 ", next(model.parameters()).device)
    model.load_state_dict(torch.load('best_weights/CASIA_128_ELA.h5'))
    patches_128 = patch_func(image, ela_image, 128)
    with torch.no_grad():
        for x in patches_128:
            trans = valid_aug(image=x[0], ela=x[1])
            image_tensor = trans["image"].unsqueeze(0).cuda()
            ela_tensor = trans["ela"].unsqueeze(0).cuda()

            _, (_, enc_out, _, _) = model(image_tensor, ela_tensor)
            res.append(enc_out.cpu().detach())
            # print(enc_out.shape)
        
    # # print("=====>3 ", next(model.parameters()).device)
    model.load_state_dict(torch.load('best_weights/CASIA_64_ELA.h5'))
    patches_64 = patch_func(image, ela_image, 64)
    with torch.no_grad():
        for x in patches_64:
            trans = valid_aug(image=x[0], ela=x[1])
            image_tensor = trans["image"].unsqueeze(0).cuda()
            ela_tensor = trans["ela"].unsqueeze(0).cuda()

            _, (_, enc_out, _, _) = model(image_tensor, ela_tensor)
            res.append(enc_out.cpu().detach())
            # print(enc_out.shape)

    del(model)
    
    return torch.cat(res, dim=0) 