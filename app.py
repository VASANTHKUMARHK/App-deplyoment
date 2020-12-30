
from flask import Flask, abort, jsonify, request, render_template
#from sklearn.externals import joblib
import joblib
import numpy as np
import json
import pandas as pd
# load the built-in model 


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/getdelay', methods=['POST','GET'])
def get_delay():
    result=request.form
    preg=result['pregnancies']
    glucose=result['glucose']
    bp=result['bloodpressure']
    st=result['skinthickness']
    insulin=result['insulin']
    bmi=result['bmi']
    dpf=result['dpf']
    age=result['age']
    print("running")
    user_input={'Pregnancies':preg,'Glucose':glucose,'BloodPressure':bp,'SkinThickness':st,'Insulin':insulin,'BMI':bmi,'DPF':dpf,'Age':age}
    log_model = joblib.load('Log.pkl')
    df=pd.DataFrame(data=user_input,index=[0])
    prediction=log_model.predict(df)
    if prediction ==1:
       return render_template('result.html')
    if prediction ==0:
       return render_template('result2.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
