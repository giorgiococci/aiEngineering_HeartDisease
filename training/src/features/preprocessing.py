import os

import numpy as np
import pandas as pd

from pickle import dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.compose import ColumnTransformer

def save_pipeline(model, filename, dataframe=None):

    basepath = os.path.join(os.path.abspath(""), "models")

    # Save pickle model
    out_file = os.path.join(basepath, filename + ".pkl")
    dump(model, open(out_file, "wb"))
    print(f"Pickle exported: {out_file}")
    return True

def preprocessing(data):

    #separate features and target as x & y
    y = data['target']
    x = data.drop('target', axis = 1)

    # Pipeline for numerical feature

    # Identify numerical columns
    numeric_features = data.select_dtypes(include=["int64", "float64"]).columns

    # Numerical Feature Pipeline
    numerical_pipeline = Pipeline (
        steps=[
            ('std_scaler', StandardScaler())
        ]
    )


    # Identify Categorical Features
    categorical_features = data.select_dtypes(include=["object"]).columns

    # Pipeline for categorical feature
    categorical_pipeline = Pipeline (
        steps=[
            ('iso_map', Isomap(n_components = 3 , n_neighbors = 6))
        ]
    )

    # Merge pipelines
    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numerical_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    final_pipeline.fit(data)
    result = final_pipeline.transform(data)

    # Save final pipeline
    save_pipeline(final_pipeline, "PipelineFeatureEngineering", data)

    return result, y
