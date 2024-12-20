
import streamlit as st
import numpy as np
import pandas as pd
import os
import io
from google.cloud import aiplatform
import google.generativeai as genai
from sklearn.preprocessing import StandardScaler
import joblib  # For loading your trained model

model_diabetes = joblib.load('diabetes_predictor.pkl') 

genai.configure(api_key="AIzaSyCes7jo5QiuKmhVTwVHNedLItNdMMlPlCY")
model = genai.GenerativeModel("gemini-1.5-flash")


def make_prediction(input_data):
    prediction = model_diabetes.predict([input_data])
    return prediction[0]  

def get_recommendations(prediction):
    # Define prompts based on prediction result
    if prediction == 1:
        prompt = "You are an AI providing health advice. Give the disclaimer about that and that and remind them to consult a healthcare provider for more information. Tell the patient they are at high risk of diabetes and offer them detailed and practical recommendations for it, including diet, exercise, and stress management tips."
    else:
        prompt = "You are an AI providing health advice. Give the disclaimer about that and that and remind them to consult a healthcare provider for more information. Tell the patient they are not at high risk of diabetes, so offer them general health maintenance recommendations, including healthy habits for diet, exercise, and mental well-being."
    response = model.generate_content(prompt)
    return response.text 

st.title('🩺 Diabetes prediction Application')
st.subheader('Please fill out the following questionnaire to obtain information about your health!')

with st.sidebar.expander("ℹ️ About this App"):
    st.write("""
    - **Purpose**: This app predicts your diabetes risk and provides actionable recommendations.
    - **How It Works**: The prediction is based on health metrics such as BMI, blood pressure, and activity level.
    - **Disclaimer**: The results are not a substitute for professional medical advice.
    """)

col1, col2 = st.columns(2)

with col1: 
    sex = st.radio('**Sex:**', ('Male', 'Female'))
    age = st.radio('**Age:**', ('18 to 24', '25 to 29', '30 to 34', '35 to 39', '40 to 44', '45 to 49', '50 to 54', '55 to 59', '60 to 64', '65 to 69', '70 to 74', '75 to 79', '80 or older'))
    education = st.radio('**What is the highest grade or year of school you completed?:**', ('Never attended school or only kindergarten', 'Elementary school', 'Some high school', 'High school graduate', 'Some college or technical school', 'College graduate'))
    income = st.radio('**How is your annual household income?:**', ('Less than $10,000', 'Less than $15,000', 'Less than $20,000', 'Less than $25,000', 'Less than $35,000', 'Less than $50,000', 'Less than $75,000', '$75,000 or more'))
    highbp = st.radio('**Do you have high blood pressure?:**', ('No', 'Yes'))
    highchol = st.radio('**Have you ever been told by a health professional that your blood cholesterol is high?:**', ('No', 'Yes'))
    cholcheck = st.radio('**Have you had a cholesterol check within past 5 years?:**', ('No', 'Yes'))
    bmi = st.number_input('**Body mass index:**', min_value=10.0, max_value=50.0, step=0.1)
    smoker = st.radio('**Have you smoked at least 100 cigarettes in your entire life?:**', ('No', 'Yes'))

with col2:
    stroke = st.radio('**Have you ever been told by a health professional that you had a stroke?:**', ('No', 'Yes'))
    heartdisease = st.radio('**Have you ever been reported having coronary heart disease (CHD) or myocardial infarction (MI)?:**', ('No', 'Yes'))
    physactivity = st.radio('**Have you had physical activity or exercise during the past 30 days other than your regular job?:**', ('No', 'Yes'))
    fruits = st.radio('**Do you consume at least one fruit per day?:**', ('No', 'Yes'))
    veggies = st.radio('**Do you consume vegetables one or more times per day?:**', ('No', 'Yes'))
    alcohol = st.radio('**Are you an adult men having more than 14 drinks per week or adult women having more than 7 drinks per week?:**', ('No', 'Yes'))
    healthcare = st.radio('**Do you have any kind of health care coverage?:**', ('No', 'Yes'))
    docbccost = st.radio('**Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?:**', ('No', 'Yes'))
    genhlth = st.radio('**Would you say that in general your health is:**', ('Excellent', 'Very good', 'Good', 'Fair', 'Poor'))
    menthlth = st.slider('**Now thinking about your mental health, for how many days during the past 30 days was it not good?:**', min_value=0, max_value=30, step=1)
    physhlth = st.slider('**Now thinking about your physical health, for how many days during the past 30 days was it not good?:**', min_value=0, max_value=30, step=1)
    diffwalk = st.radio('**Do you have serious difficulty walking or climbing stairs?:**', ('No', 'Yes'))


def yes_no_to_binary(value):
    return 1 if value == 'Yes' else 0

# Change response to numeric value
if genhlth == 'Excellent':
    genhlth = 1
elif genhlth == 'Very good':
    genhlth = 2
elif genhlth == 'Good':
    genhlth = 3
elif genhlth == 'Fair':
    genhlth = 4
else:
    genhlth = 5 

if age == '18 to 24':
    age = 1
elif age == '25 to 29':
    age = 2
elif age == '30 to 34':
    age = 3
elif age == '35 to 39':
    age = 4
elif age == '40 to 44':
    age = 5
elif age == '45 to 49':
    age = 6
elif age == '50 to 54':
    age = 7
elif age == '55 to 59':
    age = 8
elif age == '60 to 64':
    age = 9
elif age == '65 to 69':
    age = 10
elif age == '70 to 74':
    age = 11
elif age == '75 to 79':
    age = 12
elif age == '80 or older':
    age = 13

if education == 'Never attended school or only kindergarten':
    education = 1
elif education == 'Elementary school':
    education = 2
elif education == 'Some high school':
    education = 3
elif education == 'High school graduate':
    education = 4
elif education == 'Some college or technical school':
    education = 5
elif education == 'College graduate':
    education = 6

if income == 'Less than $10,000':
    income = 1
elif income == 'Less than $15,000':
    income = 2
elif income == 'Less than $20,000':
    income = 3
elif income == 'Less than $25,000':
    income = 4
elif income == 'Less than $35,000':
    income = 5
elif income == 'Less than $50,000':
    income = 6
elif income == 'Less than $75,000':
    income = 7
elif income == '$75,000 or more':
    income = 8


# Button to trigger prediction
if st.button('Predict'):   
    inputs = [
        yes_no_to_binary(highbp),
        yes_no_to_binary(highchol),
        yes_no_to_binary(cholcheck),
        bmi,
        yes_no_to_binary(smoker),
        yes_no_to_binary(stroke),
        yes_no_to_binary(heartdisease),
        yes_no_to_binary(physactivity),
        yes_no_to_binary(fruits),
        yes_no_to_binary(veggies),
        yes_no_to_binary(alcohol),
        yes_no_to_binary(healthcare),
        yes_no_to_binary(docbccost),
        genhlth,
        menthlth,
        physhlth,
        yes_no_to_binary(diffwalk),
        1 if sex == 'Male' else 0,  
        age,
        education,
        income
    ]
    
    prediction = make_prediction(inputs)
    if prediction == 1:
        st.warning('You are at high risk for diabetes.')
    else:
        st.success('Your health status is normal.')

    with st.sidebar.expander("**💡 Recommendations**"):
        recommendations = get_recommendations(prediction)
        st.write(recommendations)

    results_text = f"Prediction: {prediction}\n\nRecommendations:\n{recommendations}"
    st.sidebar.download_button(
    label="📥 Download Results",
    data=results_text,
    file_name="diabetes_results.txt",
    mime="text/plain",
)

