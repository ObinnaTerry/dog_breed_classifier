from flask import Flask, Response, request, abort, render_template, flash, redirect
from model_predict import read_all, get_prediction
import json
import os

# Create the application instance
# app = connexion.App(__name__, specification_dir='./')

# Read the swagger.yml file to configure the endpoints
# app.add_api('swagger.yml')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

UPLOAD_FOLDER = './uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_predict(data):
    if data['human']:
        return f"Hello Human... You look like a {data['prediction']}"

    elif not data['human'] and data['prediction']:
        return f"My prediction is... {data['prediction']}"

    else:
        return "I could detect neither human nor dog in your image"


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        # return render_template("result.html", output='whats up')
        img = request.files['image']
        if img and allowed_file(img.filename):
            path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            img.save(path)

            res = get_prediction(f'./uploads/{img.filename}')
            prediction = check_predict(res)

            return render_template("result.html", output=prediction)
        else:
            flash('Invalid File Type: Allowed image types are -> png, jpg, jpeg, gif', 'error')
            return redirect(request.url)


@app.route('/api/predict', methods=['POST'])
def predict():
    # img = Image.open(request.files['file'])
    # print(img)
    if 'file' not in request.files:
        return abort(404, 'Invalid upload format')

    img = request.files['image']

    if img and allowed_file(img.filename):
        path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(path)

        res = get_prediction(f'./uploads/{img.filename}')

        return Response(response=json.dumps(res), status=200)


@app.route('/api/all')
def read():
    # return "Hello {}!".format(name)
    res = read_all()
    res = json.dumps(res)
    return Response(response=res, status=200)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
