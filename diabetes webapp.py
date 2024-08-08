# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 22:31:48 2024

@author: Dell
"""

# -*- coding: utf-8 -*-
"""

Made by soumyadeep
"""

import numpy as np
import pickle
import streamlit as st


#loaded_model = pickle.load(open('D:/Work\Machine Learning/Deploying Machine Learning model/trained_model.sav', 'rb'))
loaded_model = pickle.load(open('D:\diabetes/trained_model.sav', 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    st.text('Standard is 3')
    Glucose = st.text_input('Glucose Level')
    st.text('Range:- 0-199')
    BloodPressure = st.text_input('Blood Pressure value')
    st.text('Range:- 0-122')
    SkinThickness = st.text_input('Skin Thickness value')
    st.text('Range:- 0-99')
    Insulin = st.text_input('Insulin Level')
    st.text('Range:- 0-846')
    BMI = st.text_input('BMI value')
    st.text('Range:- 0-67')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    st.text('Range:- 0.07-2.42')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    #for running directky from anaconda
    
if __name__ == '__main__':
    main()
    
    