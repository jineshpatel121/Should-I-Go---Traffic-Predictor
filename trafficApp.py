from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('Traffic/METR-LA.pkl')

# Load the historical data
historical_data = pd.read_csv('Traffic/copy.csv', index_col=0, parse_dates=True)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the route for handling form submissions
@app.route('/predict', methods=['POST'])
def predict():
    date = request.form['date']
    time = request.form['time']
    zone = int(request.form['zone'])  # Get the selected zone from the form
    
    errors = {}
    
    # Validate date
    try:
        datetime_obj = pd.to_datetime(f"{date} {time}", format='%m/%d %I:%M %p')
    except ValueError:
        errors['date'] = "Invalid date or time format. Please use MM/DD for date and HH:MM am/pm for time."
    
    if errors:
        return render_template('home.html', errors=errors, date=date, time=time, zone=zone)
    
    hour_of_day = datetime_obj.hour
    day_of_week = datetime_obj.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Select the sensors' data for the given zone
    start_sensor = (zone - 1) * 23
    end_sensor = start_sensor + 23
    sensors_data = historical_data.iloc[:, start_sensor:end_sensor]
    
    # Calculate the average speed for the selected sensors
    historical_data['speed'] = sensors_data.mean(axis=1)
    
    # Extract historical data for the same sensor and compute lag features and rolling mean
    historical_data['hour_of_day'] = historical_data.index.hour
    historical_data['day_of_week'] = historical_data.index.dayofweek
    historical_data['is_weekend'] = historical_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Filter historical data for the same hour of day, day of week, and weekend status
    relevant_data = historical_data[
        (historical_data['hour_of_day'] == hour_of_day) &
        (historical_data['day_of_week'] == day_of_week) &
        (historical_data['is_weekend'] == is_weekend)
    ]
    
    # Check if there are enough rows to compute lag features and rolling mean
    if len(relevant_data) >= 3:
        # Compute lag features and rolling mean
        relevant_data['lag_1'] = relevant_data['speed'].shift(1)
        relevant_data['lag_2'] = relevant_data['speed'].shift(2)
        relevant_data['rolling_mean_3'] = relevant_data['speed'].rolling(window=3).mean()

        # Get the latest values for lag features and rolling mean
        latest_data = relevant_data.iloc[-1]
        lag_1 = latest_data['lag_1']
        lag_2 = latest_data['lag_2']
        rolling_mean_3 = latest_data['rolling_mean_3']
    else:
        # Handle the case where there are not enough rows
        lag_1 = 0  # Placeholder value
        lag_2 = 0  # Placeholder value
        rolling_mean_3 = 0  # Placeholder value

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'hour_of_day': [hour_of_day],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'lag_1': [lag_1],
        'lag_2': [lag_2],
        'rolling_mean_3': [rolling_mean_3]
    })
    
    # Make predictions using the model
    prediction = model.predict(input_data)[0]

    # Map the prediction to a traffic level
    if prediction > 65:
        result = "Zero Traffic"
    elif prediction > 50:
        result = "Low Traffic"
    elif prediction > 40:
        result = "Moderate Traffic"
    elif prediction > 30:
        result = "High Traffic"
    else:
        result = "Extreme Traffic"    

    return render_template('home.html', result = result, date=date, time=time, zone=zone)

if __name__ == '__main__':
    app.run(debug=True)