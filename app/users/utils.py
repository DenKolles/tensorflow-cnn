import secrets
import os
from PIL import Image
from flask import current_app


def save_avatar(form_avatar):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_avatar.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(current_app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_avatar)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn
