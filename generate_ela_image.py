import os
from PIL import Image
from PIL import Image, ImageChops

def ELA(param, ROOT):
    """Performs Error Level Analysis over a directory of images"""
    
    img_path = param[0]
    ela_path = param[-1]

    TEMP = os.path.join(ROOT, img_path[:-4]+'ela_' + 'temp.jpg')
    SCALE = 25
    original = Image.open(os.path.join(ROOT, img_path))
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)
        
    except:
        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)
        
       
    d = diff.load()
    
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    diff.save(os.path.join(ROOT,ela_path))
    os.remove(TEMP)