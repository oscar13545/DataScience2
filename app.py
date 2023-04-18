import pandas as pd
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = joblib.load("ExtraTrees.pkl")

def main():
    st.title('Titanic Survival Prediction')
    col1, col2 = st.columns(2)



    with col1:
        passenger_id = st.number_input('Passenger ID', min_value=1, max_value=999999, value=1)
        pclass = st.selectbox('Pclass', [1, 2, 3])
        sex = st.selectbox('Sex', ['male', 'female'])
        age = st.number_input('Age', min_value=0, max_value=150, value=25)
        sibsp = st.number_input('Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)

    with col2:
        parch = st.number_input('Parents/Children Aboard', min_value=0, max_value=10, value=0)
        ticket = st.text_input('Ticket', '')
        fare = st.number_input('Fare', min_value=0, max_value=1000, value=10)
        cabin = st.text_input('Cabin', '')
        embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

    if(sex == 'male'):
        input_data = {'Pclass': pclass, 'Sex': sex, 'SibSp': sibsp, 'Parch': parch, 'Sex_female': False}
    else:
        input_data = {'Pclass': pclass, 'Sex': 'male', 'SibSp': sibsp, 'Parch': parch, 'Sex_female': True}

    # Function to preprocess input data
    def preprocess_input(input_data):
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, columns=['Sex'])
        return input_df

    if st.button('Predict'):
        input_df = preprocess_input(input_data)
        prediction = model.predict(input_df)[0]

        if prediction == 0:
            st.write('The passenger is not likely to survive.')
        else:
            st.write('The passenger is likely to survive.')

if __name__ == '__main__':
    main()