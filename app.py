import numpy as np
import pickle
import streamlit as st


# Loading the model
loaded_model = pickle.load(open('model.sav', 'rb'))

# Creating a function for prediction
def diabetes_prediction(input_data):

    # Changing the input data to numpy array
    array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    

def main():

    # giving a title
    st.title('Diabetes Prediction Web App')

    st.write("Please enter the following details to predict Diabetes:")

    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies', placeholder="e.g. 2")
    Glucose = st.text_input('Glucose level', placeholder="e.g. 120")
    BloodPressure = st.text_input('Blood Pressure value', placeholder="e.g. 70")
    SkinThickness = st.text_input('Skin Thickness value', placeholder="e.g. 20")
    Insulin = st.text_input('Insulin Level', placeholder="e.g. 80")
    BMI = st.text_input('BMI value', placeholder="e.g. 25.3")
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value', placeholder="e.g. 0.5")
    Age = st.text_input('Age of the Person', placeholder="e.g. 30")

    # Prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diagnosis = diabetes_prediction(input_data)

    st.success(diagnosis)



if __name__ == '__main__':
    main()
