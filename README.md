# Email Spam Classifier

This project implements a machine learning-based email spam classifier using Python. It utilizes natural language processing (NLP) techniques and various classification algorithms to distinguish between spam and non-spam (ham) emails.

## Features

- **Data Preprocessing**: Cleans and preprocesses email text data, including tokenization, stopword removal, and stemming.
- **Feature Extraction**: Converts text data into numerical features using TF-IDF vectorization.
- **Model Training**: Trains multiple classifiers such as Support Vector Machine (SVM), Random Forest, and Naive Bayes.
- **Evaluation**: Evaluates model performance using accuracy, precision, and confusion matrix.
- **Visualization**: Provides visual insights through plots like word clouds and bar graphs.
- **Prediction**: Allows prediction of new email texts to classify them as spam or ham.


## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Miheret-Girmachew/email-spam-classifier.git
   cd email-spam-classifier
   python app.py
   pip install -r requirements.txt
   streamlit run app.py
  ```

