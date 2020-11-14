import streamlit as st

from src.data.load import load_sklearn_object
from src.menu import side_menu

def main_section():
    st.title("Dashboard for Heart Disease Prediction")
    st.text("In this dashboard you can predict if a patient could have a heart disease.")



if __name__ == "__main__":


    model = load_sklearn_object("RandomForest.pkl") 
    feature_eng = load_sklearn_object("featureTransformations.pkl")

    try:
        main_section()
    except Exception as e:
        print(f'Error during streamlit launch: {e}')