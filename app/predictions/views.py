import os
from flask import render_template, url_for, flash, redirect, abort, current_app
from flask_login import current_user, login_required
from app import db
from app.predictions.forms import PredictionForm
from app.predictions.utils import save_prediction_image
from app.models import Prediction
from app.predictOne import predict_image

from flask import Blueprint

predictions = Blueprint('predictions', __name__)


@predictions.route('/predict/new', methods=['GET', 'POST'])
@login_required
def predict():
    form = PredictionForm()
    if form.validate_on_submit():
        for image in form.image.data:
            if image:
                save_prediction_image(image)
                image_path = os.path.join(current_app.root_path, 'static/prediction_images', image.filename)
                prediction = predict_image(image_path)
                full_prediction = Prediction(image_file=image.filename,
                                             user_id=current_user.id,
                                             class_id=prediction['class_id'],
                                             probability=prediction['probability'])
                db.session.add(full_prediction)
        db.session.commit()
        return redirect(url_for('main.index'))
    return render_template('predict.html', title='Predict', form=form)


@predictions.route("/predict/<int:prediction_id>")
def get_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    return render_template('prediction.html', prediction=prediction)


@predictions.route("/predict/<int:prediction_id>/delete", methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    if prediction.author != current_user:
        abort(403)
    db.session.delete(prediction)
    db.session.commit()
    return redirect(url_for('main.index'))


@predictions.route("/predict/delete", methods=['POST'])
@login_required
def delete_predictions():
    predictions = current_user.predictions
    for prediction in predictions:
        db.session.delete(prediction)
    db.session.commit()
    flash('Your predictions have been deleted!', 'success')
    return redirect(url_for('main.index'))
