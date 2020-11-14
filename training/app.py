import os
import pandas as pd

from src.features import preprocessing
from src.models import train_model

if __name__ == "__main__":

    BASEPATH = os.path.abspath("data")
    DATASET = "heart.csv"

    dataset_path = os.path.join(BASEPATH, DATASET)

    df = pd.read_csv(dataset_path, sep=',', low_memory=False)

    # Feature Engineering
    X, y = preprocessing.preprocessing(df)

    # Split dataset and train model
    dataset_splitted = train_model.split_dataset(X, y)
    train_model.train(dataset_splitted)