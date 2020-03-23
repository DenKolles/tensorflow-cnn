from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from wtforms import SubmitField, MultipleFileField


class PredictionForm(FlaskForm):
    image = MultipleFileField('Upload images for a prediction on each one', validators=[FileAllowed(['jpg', 'png'])])
    submit = SubmitField('Predict')