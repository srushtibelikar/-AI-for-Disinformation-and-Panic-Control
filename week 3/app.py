from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        return render_template('index.html', prediction_text=f'Result: {prediction}')
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
