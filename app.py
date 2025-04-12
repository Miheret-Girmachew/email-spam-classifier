import streamlit as st
import pickle

# Load the pre-trained model and the TF-IDF vectorizer
with open("logistic_regression.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("feature_extraction.pkl", "rb") as fe_file:
    feature_extraction = pickle.load(fe_file)

# Set the title of the web app
st.title("Email Spam Classifier")

# Write a brief description
st.write("Enter the content of an email below, and the classifier will predict whether it is spam or ham.")

# Text area for input email content
email_text = st.text_area("Email Content", height=250)

# Button to trigger prediction
if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some text for the email!")
    else:
        # Transform the input email text using the loaded TF-IDF vectorizer
        input_features = feature_extraction.transform([email_text])
        
        # Use the model to make a prediction
        prediction = model.predict(input_features)
        
        # Display the result
        if prediction[0] == 1:
            st.success("The email is predicted as: **Ham**")
        else:
            st.error("The email is predicted as: **Spam**")
