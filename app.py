from copyreg import pickle
import streamlit as st
import numpy as np
from numpy import array
from numpy import argmax
from numpy import genfromtxt
import pandas as pd
import shap
import xgboost as xgb  ###xgboost
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Probability prediction of hypoxemia during endoscopies", layout="wide")

plt.style.use('default')

df=pd.read_csv('x_train_web.csv',encoding='utf8')
trainy=df.hypoxemia
trainx=df.drop('hypoxemia',axis=1)
xgb = XGBClassifier(colsample_bytree=0.5,gamma=1,learning_rate=0.01,max_depth=2,
                    n_estimators =300,min_child_weight=1,subsample=0.5,
                    objective= 'binary:logistic',random_state = 1)
xgb.fit(trainx,trainy)

#mlp = MLPClassifier(hidden_layer_sizes=(80,), alpha=1e-4,activation='identity',
                    #learning_rate='invscaling',power_t=0.5,
                    #solver='lbfgs', verbose=10, tol=1e-4,
                    #learning_rate_init=0.1,random_state=1)
#mlp=mlp.fit(trainx, trainy)


def user_input_features():
    st.title("Probability prediction of hypoxemia")
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.number_input("BMI",min_value=10.0,max_value=None,step=0.1)
    a2 = st.sidebar.selectbox('OSAHS',('NO','YES'))
    a3 = st.sidebar.slider('Basal_oxygen_saturation', 90.0, 100.0, 95.0,1.0)
    a4 = st.sidebar.number_input("Remifentanil_dosage",min_value=0.1,max_value=None,step=0.1)
    
    result=""
    if a2=="Yes":
        a2=1
    else: 
        a2=0 
    output = [a1,a2,a3,a4]
    return output

outputdf = user_input_features()
outputdf = pd.DataFrame([outputdf], columns= trainx.columns)
#from sklearn import preprocessing
#lbl = preprocessing.LabelEncoder()
#outputdf['OSAHS'] = lbl.fit_transform(outputdf['OSAHS'].astype(str))


p1 = xgb.predict(outputdf)[0]
p2 = xgb.predict_proba(outputdf)

p3 = p2[:,1]
result=""
if st.button("Predict"):
  #st.write(p2)  
  #st.write(f'The probability of hypoxemia during endscopies: {p3*100}')
  st.success('The probability of hypoxemia during endscopies: {:.2f}%'.format(p3[0]*100))
  #if p3 > 0.217:
      #b="High risk"
  #else:
      #b="Low risk"
  #st.success('The risk group:'+ b)
  
  explainer = shap.TreeExplainer(xgb.predict_proba,trainx)
  shap_values = explainer.shap_values(outputdf)
  shap.plots.waterfall(shap_values[0])

#from shap.plots import _waterfall
#st_shap(shap.plots.waterfall(shap_values[0]),  height=500, width=1700)
  st.set_option('deprecation.showPyplotGlobalUse', False)
  _waterfall.waterfall_legacy(explainer.expected_value,shap_values[0,:],feature_names=trainx.columns)
#shap.summary_plot(shap_values,outputdf,feature_names=X.columns)
  st.pyplot(bbox_inches='tight')
  

#p3 = p2[:,1]
#result=""
#if st.button("Predict"):
  #st.write(p2)  
  #st.write(f'The probability of hypoxemia during endscopies: {p3*100}')
  #st.success('The probability of hypoxemia during endscopies: {:.2f}%'.format(p3[0]*100))
#p3 = p2[:,1]*100
    #st.success(p2)
    #st.success('The probability of hypoxemia during endscopies: {:.1f}%'.format(p2*100))
    #st.write('The Gastric Volume',round(p2,2))
    #st.success(p2.reshape(1))

    
  
