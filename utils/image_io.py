from PIL import Image
import numpy as np

def pil_to_np(image):
    return np.array(image)

def np_to_pil(array):
    return Image.fromarray(array)
