import cv2
import numpy
from PIL import Image
from scipy import ndimage


def grey_scale(image):
    image = image.convert('1', dither=Image.NONE)
    return image


def clean(image):
    image = image.convert("RGBA")
    data = image.getdata()
    newData = []
    for item in data:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    image.putdata(newData)
    return image


def crop(image):
    image = image.crop(image.getbbox())
    return image


def preprocess(image):
    # convert from cv2 to PIL
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    image = grey_scale(image)
    image = clean(image)
    image = crop(image)

    # convert from PIL to cv2
    image = numpy.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def multiplyRotate(image):
    images = []
    for angle in range(36):
        images.append(ndimage.rotate(image, angle*10, reshape=False))
    return images
