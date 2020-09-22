from flask import (Flask, render_template, request)
import pandas as pd
import numpy as np
import joblib
import pickle

app = Flask(__name__)
model = joblib.load(open("crash_predictor2.pkl", "rb"))
# prediction function


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/page2')
def page2():
    return render_template('index2.html')


@app.route('/predict', methods=['POST'])
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 8)
    result = model.predict(to_predict)
    return result[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'You did not survive'
        else:
            prediction = 'YOU SURVIVED!'
        return prediction


if __name__ == '__main__':
    app.run(debug=True)
