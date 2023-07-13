from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sys

from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    try:
        logging.info(f"{20*'#'} Application.py {20*'#'}")
        logging.info("Appliation execution starterd!!")

        if request.method=='GET':
            return render_template('home.html')
        else:
            data=CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))

            )
            pred_df=data.get_data_as_data_frame()
            logging.info(f"Pred dataframe in application:\n {pred_df}")
            logging.info("Before Prediction")

            predict_pipeline=PredictPipeline()
            logging.info("Mid Prediction")

            results=predict_pipeline.predict(pred_df)
            logging.info(f"After Prediction:\n {results}")

            return render_template('home.html',results=results[0])
        
    except Exception as e:
        raise CustomException(e, sys)

if __name__=="__main__":
    app.run(host="127.0.0.1")

