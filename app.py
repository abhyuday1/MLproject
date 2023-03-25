from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

##route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            bedrooms=request.form.get('bedrooms'),
            bathrooms=request.form.get('bathrooms'),
            sqft_living=request.form.get('sqft_living'),
            floors=request.form.get('floors'),
            waterfront=request.form.get('waterfront'),
            view=request.form.get('view'),
            condition=request.form.get('condition'),
            grade=request.form.get('grade'),
            zipcode=request.form.get('zipcode'),
            year=request.form.get('year'),
            month=request.form.get('month')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=result[0])
    

if __name__==" __main":
    app.run(host="0.0.0.0",debug=True)