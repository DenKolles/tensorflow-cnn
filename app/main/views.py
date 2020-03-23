from flask import render_template, request
from flask_login import current_user
from app.models import Prediction

from flask import Blueprint

main = Blueprint('main', __name__)


@main.route('/')
@main.route('/index')
def index():
    if current_user.is_authenticated:
        page = request.args.get('page', 1, type=int)
        predictions = Prediction.query.filter(Prediction.user_id == current_user.id).order_by(
            Prediction.date_created.desc()).paginate(page=page, per_page=5)
        return render_template('index.html', predictions=predictions)
    return render_template('index.html')
