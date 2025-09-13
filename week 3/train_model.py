import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = pd.read_csv("dataset.csv")

# 2. Convert 'subject' to numeric labels
df['label'] = df['subject'].factorize()[0]  # Creates numbers 0,1,2,...

# 3. Features and target
X = df['text']
y = df['label']

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)

# 6. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 7. Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# 8. Save the vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)

print("Model and vectorizer saved successfully!")
