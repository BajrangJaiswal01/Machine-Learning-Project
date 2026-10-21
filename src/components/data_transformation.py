import sys 
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer  

# for missing values
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging


from src.utils import save_object


# defining the path fro the pkl file
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation for converting
        categorical to numerical any many more etc
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

        
    # now start the actual transformation on dataset     
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")


            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]


            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)



            """ 
             Overall conclusion: Input features ko preprocess karne ke baad target column ke 
             saath jodkar training aur testing ke liye complete arrays banaye ja rahe hain.

             Preprocessing ke baad input features NumPy array mein hain aur target alag DataFrame mein hai.
             np.c_[] dono ko column-wise jod deta hai.
             
            """
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")


            # to save the pkl file which we have created in the upper code and mainly from the utils.py
            save_object(

                # file path which we defined in the configuration
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys) 


""" 
  conclusion : 
1. Train aur test CSV read karna — pd.read_csv() se dono datasets load hote hain.

2. Preprocessor banana — get_data_transformer_object() mein numerical aur categorical
 columns ke preprocessing rules define hote hain.

3. Numerical data process karna — SimpleImputer(strategy="median") missing values ko 
median se fill karta hai aur StandardScaler() values ko scale karta hai.

4. Categorical data process karna — SimpleImputer(strategy="most_frequent") missing values 
ko most common value se fill karta hai, OneHotEncoder() categories ko numerical form mein convert 
karta hai aur StandardScaler(with_mean=False) encoded values ko scale karta hai.

5. Input aur target alag karna — math_score target hai (y), baaki columns input features hain (X).

6. Preprocessing apply karna — fit_transform() training data se preprocessing rules 
seekhta bhi hai aur apply bhi karta hai. transform() test data par wahi learned rules apply karta hai.

7. Arrays jodna aur preprocessor save karna — np.c_[] input features aur math_score ko wapas
 jodta hai. save_object() preprocessor ko .pkl file mein save karta hai, taaki baad mein reuse kar sakein.

Overall conclusion: Data Transformation raw train-test data ko clean, encode aur scale karke model
training ke liye numerical arrays mein convert karta hai aur future use ke liye 
preprocessing object save karta hai.
  
"""
        
