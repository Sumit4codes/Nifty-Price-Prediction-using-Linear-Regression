from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('model/linear_regression_model_best.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    open_price = float(request.form['open_price'])
    prev_close_price = float(request.form['prev_close_price'])
    volume = float(request.form['volume'])
    high_price = float(request.form['high_price'])
    low_price = float(request.form['low_price'])

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[open_price, prev_close_price, volume, high_price, low_price]],
                              columns=['Open', 'prev_close', 'Volume', 'High', 'Low'])
    
    # Predict the closing price
    predicted_close_price = model.predict(input_data)[0]
    
    return render_template('index.html', prediction=predicted_close_price)

if __name__ == '__main__':
    app.run(debug=True)
