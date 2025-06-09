#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 19:21:51 2020

@author: amanulla
"""

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import joblib
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["GET","POST"])
def predicts():
    if request.method == "POST":
        # Getting the Data via POST Request
        print(request.form)
        age = request.form.get('age',type=int)
        bmi = request.form.get('bmi',type=float)
        children = request.form.get('children',type=int)
        male = request.form.get('male',type=int)
        smoker = request.form.get('smoker',type=int)
        region = request.form.get('region',type=int)
        # Loading the model
        model = joblib.load('save_model/insaurance_LR.pkl')
        # Organizing the data
        x_inp = np.array([[age,bmi,children,male,smoker,region]])
        print(x_inp)
        # Predicting the data
        predicted_result = model.predict(x_inp)
        predicted_result = predicted_result * -1
        final_pred = predicted_result[0]
        result_string = 'The Insurance for the Data is '+str(final_pred)
        #Return the Prediction
        return render_template('index.html',data = result_string)


if __name__ == '__main__':
    app.run()

