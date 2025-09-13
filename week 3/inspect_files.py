import pickle

# Load and inspect the model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
    print(model)
except Exception as e:
    print("Error loading model:", e)

# Load and inspect the vectorizer
try:
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    print("\nVectorizer loaded successfully!")
    print(vectorizer)
except Exception as e:
    print("Error loading vectorizer:", e)
