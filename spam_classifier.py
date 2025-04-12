import pandas as pd
import numpy as np
import warnings
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# Load dataset
raw_mail_data = pd.read_csv("mail_data.csv")

# Fill missing values
df = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Encode labels: spam -> 0, ham -> 1
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

df['Message'] = df['Message'].apply(preprocess_text)

# Features and Labels
X = df['Message']
Y = df['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# TF-IDF Vectorization (enhanced settings)
feature_extraction = TfidfVectorizer(min_df=2, max_df=0.85, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Ensure label format
y_train = y_train.astype("int")
y_test = y_test.astype("int")

# Model training with class weight balance
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_features, y_train)

# Evaluate on training data
prediction_train_data = model.predict(X_train_features)
accuracy_train_data = accuracy_score(y_train, prediction_train_data)
print("Accuracy on train data: ", accuracy_train_data)

# Evaluate on test data
prediction_test_data = model.predict(X_test_features)
accuracy_test_data = accuracy_score(y_test, prediction_test_data)
print("Accuracy on test data: ", accuracy_test_data)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, prediction_test_data, target_names=["Spam", "Ham"]))

# Prediction for new input
input_user_mail = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]
input_user_mail = [preprocess_text(mail) for mail in input_user_mail]
input_data_features = feature_extraction.transform(input_user_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("This is a ham mail")
else:
    print("This is a spam mail")

# Save model and vectorizer
pickle.dump(model, open("logistic_regression.pkl", "wb"))
pickle.dump(feature_extraction, open("feature_extraction.pkl", "wb"))
