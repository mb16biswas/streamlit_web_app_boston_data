import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import datasets
# from sklearn.ensemble import RandomForestClassifier
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])
load_clf = pickle.load(open('boston_model.pkl', 'rb'))

st.write("""
# Boston house price prediction 

Data obtained from the https://www.kaggle.com/c/boston-housing/data
""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    
    input_df.drop(input_df.columns[input_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    #https://stackoverflow.com/questions/43983622/remove-unnamed-columns-in-pandas-dataframe#:~:text=First%2C%20find%20the%20columns%20that,drop%20parameters%20as%20well.
    
    st.subheader('Prediction')
    st.write(load_clf.predict(input_df))

else:
    def user_input_features():


                    CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
                    ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
                    INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
                    CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
                    NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
                    RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
                    AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
                    DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
                    RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
                    TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
                    PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
                    B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
                    LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
                    data = {'CRIM': CRIM,
                            'ZN': ZN,
                            'INDUS': INDUS,
                            'CHAS': CHAS,
                            'NOX': NOX,
                            'RM': RM,
                            'AGE': AGE,
                            'DIS': DIS,
                            'RAD': RAD,
                            'TAX': TAX,
                            'PTRATIO': PTRATIO,
                            'B': B,
                            'LSTAT': LSTAT}

                    features = pd.DataFrame(data, index=[0])
                   
                    return features
    input_df = user_input_features()
    print(input_df)



    st.subheader('Prediction')
    st.write(load_clf.predict(input_df))


