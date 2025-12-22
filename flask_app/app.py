from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import logging
from tensorflow.keras.models import load_model, model_from_json
import pandas as pd
from pathlib import Path


app = Flask(__name__)
# Portable path to the dataset inside the flask_app folder
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'dataset.csv'
df = pd.read_csv(DATASET_PATH)
logging.basicConfig(level=logging.INFO)

# Load model and scalers
def safe_load_h5_model(h5_path):
    """Try to load an HDF5 Keras model. If the model config contains
    legacy keys (like `time_major`) not supported by the current Keras,
    remove them from the saved config and rebuild the model, then load
    weights from the file.
    """
    try:
        return load_model(h5_path)
    except ValueError as e:
        err = str(e)
        # Detect known incompatibility around `time_major` in LSTM config
        if 'time_major' not in err:
            raise

        import h5py, json, logging

        logging.warning("load_model failed due to legacy config; attempting compatibility load: %s", err)

        with h5py.File(h5_path, 'r') as f:
            model_config_raw = f.attrs.get('model_config')
            if model_config_raw is None:
                raise
            if isinstance(model_config_raw, (bytes, bytearray)):
                model_config = json.loads(model_config_raw.decode('utf-8'))
            else:
                model_config = json.loads(model_config_raw)

        # Recursively remove `time_major` from any layer config
        def strip_time_major(obj):
            if isinstance(obj, dict):
                if 'time_major' in obj:
                    obj.pop('time_major', None)
                for v in obj.values():
                    strip_time_major(v)
            elif isinstance(obj, list):
                for item in obj:
                    strip_time_major(item)

        strip_time_major(model_config)

        # Rebuild model from cleaned config and load weights
        # model_from_json expects a JSON string; provide common keras classes
        import json
        import tensorflow as tf

        custom_objects = {
            'Sequential': tf.keras.models.Sequential,
            'InputLayer': tf.keras.layers.InputLayer,
            'LSTM': tf.keras.layers.LSTM,
            'Dense': tf.keras.layers.Dense,
            'GlorotUniform': tf.keras.initializers.GlorotUniform,
            'Orthogonal': tf.keras.initializers.Orthogonal,
            'Zeros': tf.keras.initializers.Zeros,
        }

        model = model_from_json(json.dumps(model_config), custom_objects=custom_objects)
        model.load_weights(h5_path)
        return model


model = safe_load_h5_model('saved_data/lstm_model.h5')
scaler_X = joblib.load('saved_data/scaler_X.pkl')
scaler_y = joblib.load('saved_data/scaler_y.pkl')
commodity_names = joblib.load('saved_data/commodity_names.pkl')
data = pd.read_csv(DATASET_PATH)

EXPECTED_FEATURE_COUNT = scaler_X.n_features_in_


@app.route('/')
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict_page", methods=['GET', 'POST'])
def predict_page():
    return render_template("index.html", commodities=commodity_names)

@app.route('/api/commodities')
def get_commodities():
    # Read from the tomato-only dataset using a portable relative path
    df = pd.read_csv(DATASET_PATH)
    commodities = df['Commodity'].unique().tolist()
    # Ensure only tomato varieties are returned
    tomato_commodities = [c for c in commodities if 'Tomato' in c or 'tomato' in c]
    return jsonify(tomato_commodities)

@app.route("/analyze")
def analyze():
    # Get only tomato commodities
    commodities = data['Commodity'].unique().tolist()
    tomato_commodities = [c for c in commodities if 'Tomato' in c or 'tomato' in c]
    return render_template('analyze.html', commodities=tomato_commodities)

@app.route("/commodity_data/<commodity_name>")
def get_commodity_data(commodity_name):
    specific_data = data[data['Commodity'] == commodity_name]
    return jsonify({
        "dates": specific_data['Date'].astype(str).tolist(),
        "min_prices": specific_data['Minimum'].tolist(),
        "max_prices": specific_data['Maximum'].tolist()
    })

@app.route("/search_commodity")
def search_commodity():
    term = request.args.get('term')
    # Filter to only tomato varieties
    tomato_names = [com for com in commodity_names if 'Tomato' in com or 'tomato' in com]
    search_results = [com for com in tomato_names if term.lower() in com.lower()]
    return jsonify(search_results)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # Get the data from the form
        date = request.form['date']  # Format: 'YYYY-MM-DD'
        year, month, day = map(float, date.split('-'))
        commodity = request.form['commodity']

        # Debug: Check commodity count and model input shape
        logging.info(f"Total commodities: {len(commodity_names)}")
        logging.info(f"Model input shape: {model.input_shape[2]}")
        logging.info(f"One-hot vector size: {len(generate_one_hot_vector(commodity, commodity_names))}")

        # One-hot encode the commodity
        commodity_data = generate_one_hot_vector(commodity, commodity_names)

        # Debug: Check feature counts and commodity data
        logging.info(f"Commodity Input:  {commodity}")
        logging.info(f"One-hot Encoded:  {commodity_data}")
        logging.info(f"Expected Features by Scaler:  {EXPECTED_FEATURE_COUNT}")
        logging.info(f"Provided Feature Count:  {len(commodity_data) + 3}")

        # Quick fix: if features are less, pad them
        feature_count = len(commodity_data) + 3
        if feature_count < EXPECTED_FEATURE_COUNT:
            logging.warning("PADDING INPUT: Input features less than expected. Adding dummy features.")
            commodity_data.extend([0] * (EXPECTED_FEATURE_COUNT - feature_count))

        # Verify feature count
        assert len(
            commodity_data) + 3 == EXPECTED_FEATURE_COUNT, f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(commodity_data) + 3}"

        # Prepare and scale the input data
        input_data = np.array([year, month, day] + commodity_data).reshape(1, -1)

        # Debug: Check shaped input data
        logging.info(f"Input Data:  {input_data}")

        # Convert to DataFrame with feature names to avoid sklearn UserWarning
        if hasattr(scaler_X, 'feature_names_in_'):
            input_df = pd.DataFrame(input_data, columns=scaler_X.feature_names_in_)
        else:
            # Fallback if feature names are not available (e.g. older sklearn or different scaler)
            input_df = input_data

        input_scaled = scaler_X.transform(input_df)
        input_scaled = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_original = scaler_y.inverse_transform(prediction).flatten()

        result = {
            'min_price': prediction_original[0],
            'max_price': prediction_original[1]
        }

        # Send additional data to the template
        return render_template("result.html", result=result, commodity_name=commodity, date=date)
    else:
        return "Error: Method not allowed"



def generate_one_hot_vector(input_category, all_possible_categories):
    return [1 if cat == input_category else 0 for cat in all_possible_categories]


if __name__ == "__main__":
    app.run(debug=True)

