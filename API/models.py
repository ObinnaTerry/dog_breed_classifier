from datetime import datetime
from config import db, ma
from marshmallow_sqlalchemy import ModelSchema


class Prediction(db.Model):
    __tablename__ = "predictions"
    prediction_id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.String(32))
    length = db.Column(db.String(32))
    human = db.Column(db.Integer)
    timestamp = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class PredictionSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Prediction
        # load_instance = True

        sqla_session = db.session
