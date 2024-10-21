import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self):
        # Define paths for the artifacts
        self.artifacts_dir = 'artifacts'
        self.raw_data_path = os.path.join(self.artifacts_dir, "data.csv")
        self.train_data_path = os.path.join(self.artifacts_dir, "train.csv")
        self.test_data_path = os.path.join(self.artifacts_dir, "test.csv")

    def initiate_data_ingestion(self):
        """
        Reads data from the CSV file, splits it into train and test sets,
        saves the datasets to the 'artifacts' folder, and returns the DataFrames.
        """
        try:

            os.makedirs(self.artifacts_dir, exist_ok=True)

            df = pd.read_csv('E:/Machine Learning Projects/notebook/data/stud.csv')
            print("Dataframe loaded successfully.")

            df.to_csv(self.raw_data_path, index=False)
            print(f"Raw data saved at: {self.raw_data_path}")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            print(f"Train data saved at: {self.train_data_path}")
            print(f"Test data saved at: {self.test_data_path}")

            return df, train_set, test_set

        except Exception as e:
            print(f"Error in data ingestion: {e}")
            raise e

if __name__ == "__main__":
    obj = DataIngestion()
    raw_data_df, train_data_df, test_data_df = obj.initiate_data_ingestion()

