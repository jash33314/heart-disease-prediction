import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, request, render_template, redirect, url_for
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Read the CSV file into a DataFrame
df = pd.read_csv(r"C:\Users\P.P.K.JASHUVA\OneDrive\Desktop\ml\heart.csv")
# Define the target variable 'y' and the feature variables 'x'
y = df[['target']]
x = df.drop(columns=["target", "sex", "ST slope", "resting ecg", "exercise angina","fasting blood sugar","oldpeak", "chest pain type"])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Initialize the Linear Regression model
lr = LinearRegression()

# Fit the model on the training data
lr.fit(x_train, y_train)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    age = int(request.form['age'])
    resting_bp_s = int(request.form['resting_bp_s'])
    cholesterol = int(request.form['cholesterol'])
    max_heart_rate = int(request.form['max_heart_rate'])

    # Create DataFrame for prediction
    data = pd.DataFrame({
        'age': [age],
        'resting bp s': [resting_bp_s],
        'cholesterol': [cholesterol],
        'max heart rate': [max_heart_rate]
    })

    # Predict using the trained model
    prediction = lr.predict(data)

    # Redirect based on prediction
    if prediction[0][0] > 0.5:
        return redirect(url_for('tips'))
    else:
        return redirect(url_for('second'))


@app.route('/tips')
def tips():
    return render_template('tips.html')


@app.route('/second')
def second():
    return render_template('second.html')


if __name__ == "__main__":
    app.run(debug=True)
