import logging
import os

from pickle import dump

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    print("Dataset splitted")
    print(f"X_train: {X_train.shape}")
    print(f"X_train: {X_test.shape}")
    print(f"X_train: {y_train.shape}")
    print(f"X_train: {y_test.shape}")

    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    logging.debug(f"Dataset splitted: {result}")

    return result

def save_model(model, filename):

    basepath = os.path.join(os.path.abspath(""), "models/")

    out_file = os.path.join(basepath, filename + ".pkl")
    dump(model, open(out_file, "wb"))
    print(f"Model saved to: {out_file}")
    logging.debug(f"Model saved: {out_file}")
    return True
    

def train(split_dataset, model_name='DecisionTreeClassifier', max_features=None, max_depth=None):
    
    X_train = split_dataset["X_train"]
    y_train = split_dataset["y_train"]

    logging.debug(f"Start training the model...")

    model = DecisionTreeClassifier(max_features = max_features, max_depth = max_depth)

    model.fit(X_train, y_train)

    print("Training completed")
    logging.debug(f"Model trained")

    save_model(model, model_name)

    return model