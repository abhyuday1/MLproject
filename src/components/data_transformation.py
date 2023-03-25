import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import os

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''This function is responsible for the Data Transformation'''
        try:
            cat_features = ["waterfront", "view", "condition", "grade", "month","year", "zipcode"]
            num_features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors']
            
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler(with_mean=False))]
            )

            cat_pipeline=Pipeline(
                
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(sparse=False)),
                ("scalar",StandardScaler(with_mean=False))]
                
            )

           # logging.info("Month and Year Extraction completed, Date column dropped")
            logging.info("Numerical columsn standared scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_features),
                ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    

    def intiate_data_transformation(self,train_path,test_path):
        '''This function is responsible for the Data Transformation'''
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            train_df = train_df[["price","date", "bedrooms", 
                                  "bathrooms", "sqft_living", "floors", 
                                  "waterfront", "view", "condition", "grade","zipcode"]]
            test_df = test_df[["price","date", "bedrooms", 
                                  "bathrooms", "sqft_living", "floors", 
                                  "waterfront", "view", "condition", "grade","zipcode"]]
            logging.info("Feature selection is completed")

            logging.info("obtaining preprocessor onject")
            preprocessor_obj=self.get_data_transformer_object()
            
            
            train_df.loc[:,"year"] = train_df["date"].str[0:4]
            train_df.loc[:,"month"] = train_df["date"].str[4:6]
            train_df = train_df.drop(columns=["date"])
            #removing date after this extraction
            
            test_df.loc[:,"year"] = test_df["date"].str[0:4]
            test_df.loc[:,"month"] = test_df["date"].str[4:6]
            #removing date after this extraction
            test_df = test_df.drop(columns=["date"])

            categorical_col=["waterfront", "view", "condition", "grade", "month","year", "zipcode"]
            numerical_col=['bedrooms', 'bathrooms', 'sqft_living', 'floors']
            target_column_name="price"
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
           
            logging.info("Applying preprocessing on training dataframe and testing dataframe")
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            print(type(input_feature_train_arr))
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
                ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
                ]
            
             
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
           
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)