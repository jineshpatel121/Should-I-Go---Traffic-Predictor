# Should I Go?- LA Traffic Prediction

This project predicts traffic levels in Los Angeles using historical traffic data and a machine learning model. The application is built with Flask for the web interface and uses a Random Forest Regressor for the prediction model.

## Project Structure


- `METR-LA_copy.csv`: Shortened version of the orginal METR-LA dataset containing historical traffic data.
- `static/`: Directory for static files like images.
- `templates/`: Directory for HTML templates.
- `testTraffic.py`: Script for testing the traffic prediction model.
- `testTrafficLogic.py`: Script for preprocessing the data and training the model.
- `trafficApp.py`: Flask application for the web interface.
- `trafficModel.pkl`: Serialized machine learning model.

## Setup

1. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

2. Ensure you have the dataset [METR-LA_copy.csv](http://_vscodecontentref_/6) in the project directory.

3. Train the model by running:
    ```sh
    python testTrafficLogic.py
    ```

4. Start the Flask application:
    ```sh
    python trafficApp.py
    ```

5. Open your web browser and go to `http://127.0.0.1:5000/` to access the application.

## Usage

1. On the home page, enter the date, time, and zone number for which you want to predict the traffic.
2. Click the "Predict" button to get the traffic prediction.
3. The application will display the predicted traffic level.

## Model Training

The model is trained using the [testTrafficLogic.py](http://_vscodecontentref_/7) script. It preprocesses the data, extracts features, and trains a Random Forest Regressor. The trained model is saved as [trafficModel.pkl](http://_vscodecontentref_/8).

## Testing

You can test the model using the [testTraffic.py](http://_vscodecontentref_/9) script. It loads the dataset, preprocesses it, and evaluates the model's performance.

## Future

Will soon add more traffic congested cities/states (ex: Dallas, NYC, Chicago)
