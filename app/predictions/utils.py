import os
from PIL import Image
from flask import current_app


def save_prediction_image(form_image):
    picture_path = os.path.join(current_app.root_path, 'static/prediction_images', form_image.filename)
    i = Image.open(form_image)
    i.save(picture_path)
    return form_image.filename
