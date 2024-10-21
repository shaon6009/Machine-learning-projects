import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for creating and returning the data transformation pipeline.
        It handles both numerical and categorical columns separately.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
                    ("scaler", StandardScaler())  # Scale numerical values
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle missing values
                    ("one_hot_encoder", OneHotEncoder()),  # Convert to one-hot encoding
                    ("scaler", StandardScaler(with_mean=False))  # Scale categorical features
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Data transformer object created successfully.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error in creating data transformer object: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function initiates the data transformation process. It reads the train and test datasets,
        applies the preprocessing, and saves the preprocessor object.
        """
        try:
            logging.info("Reading train and test data.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying the preprocessor to the datasets.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
  
            artifacts_dir = os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path)
            logging.info(f"Creating artifacts directory at: {artifacts_dir}")
            os.makedirs(artifacts_dir, exist_ok=True)

            logging.info(f"Saving preprocessor object to: {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            if os.path.exists(self.data_transformation_config.preprocessor_obj_file_path):
                logging.info("Preprocessor object saved successfully.")
            else:
                logging.error("Failed to save preprocessor object. File not found after saving.")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)
