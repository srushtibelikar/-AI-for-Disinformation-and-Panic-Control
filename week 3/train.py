import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
df = pd.read_csv('dataset.csv')  # Replace with your dataset path

# Features and labels
X = df['text']      # The text data
y = df['subject']   # The target variable (news category)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train the model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train_tfidf, y_train)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model and vectorizer saved successfully!")
