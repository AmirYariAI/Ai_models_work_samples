import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

import jinja2 
from pycaret.classification import *

st.write("""
# Simple churn
""")

st.sidebar.header('User Input Parameters')



def user_input_features():

    CreditScore     = st.sidebar.slider('CreditScore', 350., 850., 400.)
    Age             = st.sidebar.slider('Age', 18, 92, 25)
    Tenure          = st.sidebar.slider('Tenure', 0, 10, 5)
    Balance         = st.sidebar.slider('Balance', 0., 250898.09, 97198.540)
    NumOfProducts   = st.sidebar.slider('NumOfProducts', 1, 4, 2)
    EstimatedSalary = st.sidebar.slider('EstimatedSalary', 11.58, 199992.48, 149388.2475)
    HasCrCard       = st.sidebar.slider('HasCrCard', 0, 1, 1)
    IsActiveMember  = st.sidebar.slider('IsActiveMember', 0, 1, 1)
    Gender          = st.sidebar.radio('Gender',['Male','Female'])
    Geography       = st.sidebar.radio('Geography',['Spain','France','Germany'])

    data = {'CreditScore'     : CreditScore,
            'Geography'       : Geography,
            'Gender'          : Gender,
            'Age'             : Age,
            'Tenure'          : Tenure,
            'Balance'         : Balance,
            'NumOfProducts'   : NumOfProducts,
            'HasCrCard'       : HasCrCard,
            'IsActiveMember'  : IsActiveMember,
            'EstimatedSalary' : EstimatedSalary
            }

    features = pd.DataFrame(data, index=[0])

    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

model = load_model('churn_cl_model')

st.subheader('model prediction :')

prediction = model.predict(df)

st.write(prediction)

st.write('0 means not exited ')
st.write('1 means exited')
#iris = datasets.load_iris()
#X = iris.data
#Y = iris.target

#clf = RandomForestClassifier()
#clf.fit(X, Y)

#prediction = clf.predict(df)
#prediction_proba = clf.predict_proba(df)

#st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)

#st.subheader('Prediction')
#st.write(iris.target_names[prediction])
##st.write(prediction)

#st.subheader('Prediction Probability')
#st.write(prediction_proba)

#st.balloons()

#import time
#my_bar = st.progress(0)
#for percent_complete in range(100):
#    time.sleep(0.1)
#    my_bar.progress(percent_complete + 1)

