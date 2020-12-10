from PIL import Image
import os, sys

path = "./TensorFlow/workspace/training_demo/images/train/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            print(im.size)
            f, e = os.path.splitext(path+item)
            print(f)
            imResize = im.resize((400,300), Image.NEAREST)
            imResize = im.rotate(-90)
            imResize.save(f + '_reshaped.jpg', 'JPEG', quality=90)

resize()