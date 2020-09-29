from albumentations.pytorch.functional import img_to_tensor
import torch
import numpy as np
import cv2

def patch_func(img, patch_size):
    d = img.shape
    patches = []
    for i in range(0, d[0], patch_size):
        for j in range(0, d[1], patch_size):
            x = i + patch_size
            y = j + patch_size
            if x > d[0] or y > d[1]:
                break
            temp = img[i: x, j: y]
            temp = cv2.resize(temp, (d[0],d[1]), interpolation=cv2.INTER_AREA)
            patches.append(temp)
    return np.stack(patches)


def ensemble(model, image):
    normalize = {
        "mean": [0.42468103282400615, 0.4259826707370029, 0.38855473517307415],
        "std": [0.2744059987371694, 0.2684138285232067, 0.29527622263685294],
    }

    model.to('cuda')
    model.eval()
    
    model.load_weights('256_encoder.h5')
    image_full = img_to_tensor(image, normalize).unsqueeze(0).cuda()
    with torch.no_grad():
        tensor1, _ = model(image_full)
        del(image_full)

    model.load_weights('256_encoder.h5')
    image_128 = patch_func(image, 128)
    image_128_norm = [img_to_tensor(x, normalize) for x in image_128]
    image_128_norm = torch.from_numpy(np.stack(image_128_norm)).cuda()
    with torch.no_grad():
        tensor2, _ = model(image_128_norm)
        del(image_128_norm)
    
    model.load_weights('256_encoder.h5')
    image_64 = patch_func(image, 64)
    image_64_norm = [img_to_tensor(x, normalize) for x in image_64]
    image_64_norm = torch.from_numpy(np.stack(image_64_norm)).cuda()
    with torch.no_grad():
        tensor3, _ = model(image_64_norm)
        del(image_64_norm)

    del(model)
    
    return torch.cat([tensor1, tensor2, tensor3]) 