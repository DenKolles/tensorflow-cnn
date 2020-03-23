import os
import secrets

from PIL import Image
from flask import render_template, url_for, flash, redirect, request, abort
from flask_login import login_user, current_user, logout_user, login_required

from app import app, db, bcrypt
from app.forms import LoginForm, RegistrationForm, PredictionForm, UpdateAccountForm
from app.models import User, Prediction
from app.predictOne import predict_image, save_prediction_image


@app.route('/')
@app.route('/index')
def index():
    if current_user.is_authenticated:
        page = request.args.get('page', 1, type=int)
        predictions = Prediction.query.filter(Prediction.user_id == current_user.id).order_by(
            Prediction.date_created.desc()).paginate(page=page, per_page=5)
        return render_template('index.html', predictions=predictions)
    return render_template('index.html')


@app.route('/predict/new', methods=['GET', 'POST'])
@login_required
def predict():
    form = PredictionForm()
    if form.validate_on_submit():
        for image in form.image.data:
            if image:
                save_prediction_image(image)
                image_path = os.path.join(app.root_path, 'static/prediction_images', image.filename)
                prediction = predict_image(image_path)
                full_prediction = Prediction(image_file=image.filename,
                                             user_id=current_user.id,
                                             class_id=prediction['class_id'],
                                             probability=prediction['probability'])
                db.session.add(full_prediction)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('predict.html', title='Predict', form=form)


@app.route("/predict/<int:prediction_id>")
def get_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    return render_template('prediction.html', prediction=prediction)


@app.route("/predict/<int:prediction_id>/delete", methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    if prediction.author != current_user:
        abort(403)
    db.session.delete(prediction)
    db.session.commit()
    return redirect(url_for('index'))


@app.route("/predict/delete", methods=['POST'])
@login_required
def delete_predictions():
    predictions = current_user.predictions
    for prediction in predictions:
        db.session.delete(prediction)
    db.session.commit()
    flash('Your predictions have been deleted!', 'success')
    return redirect(url_for('index'))


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Successfully registered. You can now log in', 'success')
        return redirect(url_for('index'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_avatar(form.picture.data)
            current_user.avatar = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.avatar)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)


def save_avatar(form_avatar):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_avatar.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_avatar)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn
