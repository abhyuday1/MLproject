import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.trasform(features)
            pred=model.predict(data_scaled)

            return pred
        except Exceptionas as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,bedrooms,bathrooms,sqft_living,floors,waterfront,view,condition,grade,zipcode,year,month):
        self.bedrooms=bedrooms
        self.bathrooms=bathrooms
        self.sqft_living=sqft_living
        self.floors=floors
        self.waterfront=waterfront
        self.view=view
        self.condition=condition
        self.grade=grade
        self.zipcode=zipcode
        self.year=year
        self.month=month

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "bedrooms":[self.bedrooms], 
                "bathrooms":[self.bathrooms],
                "sqft_living":[self.sqft_living],
                "floors":[self.floors], 
                "waterfront":[self.waterfront],
                "view":[self.view],
                "condition":[self.condition],
                "grade":[self.grade],
                "zipcode":[self.zipcode],
                "year":[self.year],
                "month":[self.month],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
