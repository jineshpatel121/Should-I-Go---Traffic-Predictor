import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv('Traffic/METR-LA.csv', index_col=0, parse_dates=True)


# Extract additional features from the index (datetime)
data['hour_of_day'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Melt the dataframe to have sensor IDs as a column
data = data.reset_index().melt(id_vars=['index', 'hour_of_day', 'day_of_week', 'is_weekend'], var_name='sensor_id', value_name='speed')

# Rename the datetime column
data.rename(columns={'index': 'datetime'}, inplace=True)

# Sort by datetime and sensor_id
data.sort_values(by=['datetime', 'sensor_id'], inplace=True)

# Add lag features and rolling mean for each sensor
data['lag_1'] = data.groupby('sensor_id')['speed'].shift(1)
data['lag_2'] = data.groupby('sensor_id')['speed'].shift(2)
data['rolling_mean_3'] = data.groupby('sensor_id')['speed'].rolling(window=3).mean().reset_index(level=0, drop=True)

# Drop rows with NaN values created by lag features
data.dropna(inplace=True)

# Define features and target
X = data[['hour_of_day', 'day_of_week', 'is_weekend', 'lag_1', 'lag_2', 'rolling_mean_3']]
y = data['speed']

# Check if the dataset is empty after preprocessing
if X.empty or y.empty:
    print("The dataset is empty after preprocessing. Please check the preprocessing steps.")
else:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model with the best hyperparameters
    best_model_rf = RandomForestRegressor(
        max_depth=10,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=100,
        random_state=42
    )

    # Train the model with the best hyperparameters
    best_model_rf.fit(X_train, y_train)

    # Make predictions on the test data
    predictions_test = best_model_rf.predict(X_test)

    # Calculate R-squared and Mean Squared Error for the test data
    r2_test = r2_score(y_test, predictions_test)
    mse_test = mean_squared_error(y_test, predictions_test)

    # A DataFrame to display the actual vs predicted values
    results = pd.DataFrame({'Sensor #': y_test.index, 'Actual': y_test.values, 'Predicted': predictions_test})

    results.reset_index(drop=True, inplace=True)

    print("Predicted vs Actual Values:")
    print(results.head())