from PIL import Image
import os, sys
import glob

path = "./TensorFlow/workspace/training_demo/images/test/"

def resize():
    for index, img_path in enumerate(glob.glob(path + '*.jpg')):
        img = Image.open(img_path)
        imResize = img.resize((400,300), Image.NEAREST)
        imResize = img.rotate(-90)
        filename = os.path.dirname(img_path) + ('/img_%.2d_reshped.jpg' %index)
        imResize.save(filename, 'JPEG', quality=90)

resize()