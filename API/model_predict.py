import os
import time

from models import Prediction, PredictionSchema
from predict_utils import Detector
from config import db

pred = Detector()


def db_insert(prediction, length, human):
    if not os.path.exists('predictions.db'):
        db.create_all()

    p = Prediction(prediction=prediction, length=length, human=human)
    db.session.add(p)

    db.session.commit()


def get_prediction(img_path):
    print(img_path)

    result = {'human': None, 'prediction': None}
    start_time = time.time()

    # handle cases for a human face, dog, and neither
    if pred.face_detector(img_path):
        result['human'] = True
        result['prediction'] = pred.predict(img_path)

    elif pred.dog_detector(img_path):
        result['human'] = False
        result['prediction'] = pred.predict(img_path)

    else:
        pass

    end_time = time.time() - start_time
    human = human_val(result['human'])

    db_insert(result['prediction'], str(end_time), human)

    return result


def human_val(val):
    if val:
        return 1
    elif val is None:
        return 2
    return 0


def read_all():
    preds = Prediction.query.all()
    # print(preds)
    pred_schema = PredictionSchema(many=True)

    return pred_schema.dump(preds)

# print(read_all())
# print(get_prediction('Labrador.jpg'))
# print(get_prediction('Labrador.jpg'))
