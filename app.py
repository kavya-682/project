import threading
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import streamlit as st
import requests
import os

#############################################
#          FLASK BACKEND SETUP              #
#############################################

app = Flask(__name__)

# Check if the dataset exists in the current directory.
# Make sure 'real_estate_165_data.csv' is in the same folder as app.py.
if not os.path.exists('real_estate_165_data.csv'):
    raise FileNotFoundError("The dataset file 'real_estate_165_data.csv' was not found in the current directory.")

# Load dataset
df = pd.read_csv('real_estate_165_data.csv')

target_column = 'Property_Value'
X = df.drop(target_column, axis=1)
y = df[target_column]

# Save the list of expected columns for prediction.
expected_columns = X.columns.tolist()

# Identify features for preprocessing
numerical_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = ['Location', 'Property_Type']
categorical_features = [col for col in categorical_features if col in X.columns]

# Preprocessor definition
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Split data and preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train models
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
stacking = StackingRegressor(estimators=[('rf', rf), ('gb', gb)], final_estimator=LinearRegression())

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
stacking.fit(X_train, y_train)

# Save the preprocessor and model (they will be reloaded for each prediction)
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(stacking, 'stacking_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict the property value.
    Expects a JSON payload. The training data expects the following columns:
    {}
    If some columns are missing in the input, they will be filled with a default value of 0.
    """.format(expected_columns)
    try:
        data = request.json
        df_input = pd.DataFrame([data])
        # Reindex to include all expected columns (fill missing with 0)
        df_input = df_input.reindex(columns=expected_columns, fill_value=0)
        
        # Reload the preprocessor and model
        preprocessor = joblib.load('preprocessor.pkl')
        model = joblib.load('stacking_model.pkl')
        processed_data = preprocessor.transform(df_input)
        prediction = model.predict(processed_data)[0]
        return jsonify({'predicted_value': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

def run_flask():
    """Function to run the Flask server."""
    # Disable reloader to avoid running the server twice
    app.run(port=5000, debug=False, use_reloader=False)

# Start Flask in a background thread so that Streamlit can run concurrently.
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

#############################################
#          STREAMLIT FRONTEND UI            #
#############################################

st.title("🏡 Real Estate Price Predictor")

# Input widgets for user to enter property details.
# IMPORTANT: The keys below must match (or be mapped to) the feature names used during training.
# For this example, we assume the training data includes columns such as:
# - 'Location', 'Property_Type'
# - 'Size_sqft', 'Num_Bedrooms', 'Num_Bathrooms', 'Age'
# - Additionally, training columns include 'Distance_to_City_Center', 'Property_ID', 'Neighborhood_Rating', 'Year_Built'
#
# The missing columns are now supplied with default values.
location = st.selectbox("Select Location", ["New York", "Los Angeles", "Chicago", "Houston"])
property_type = st.selectbox("Select Property Type", ["Apartment", "House", "Villa"])
area = st.number_input("Enter Area (sq ft)", min_value=100, max_value=10000, step=50)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=10, step=1)
age = st.number_input("Enter Property Age (years)", min_value=0, max_value=100, step=1)

if st.button("Predict Price"):
    # Prepare the input data for the prediction API.
    # Map the UI inputs to the expected feature names.
    # For the columns not captured by user input, we provide default values (here, 0).
    input_data = {
        "Location": location,
        "Property_Type": property_type,
        # Use the keys that match your training dataset.
        "Size_sqft": area,           # Replace with the appropriate column name if needed.
        "Num_Bedrooms": bedrooms,
        "Num_Bathrooms": bathrooms,
        "Age": age,
        # The following columns were missing. They are added with default values.
        "Distance_to_City_Center": 0,
        "Property_ID": 0,
        "Neighborhood_Rating": 0,
        "Year_Built": 0
    }
    
    try:
        # Send the request to the Flask API
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            if 'predicted_value' in result:
                prediction = result["predicted_value"]
                st.success(f"Estimated Property Value: ${prediction:,.2f}")
            else:
                st.error("Prediction error: " + str(result.get('error', 'Unknown error')))
        else:
            st.error("Error in API request. Status Code: " + str(response.status_code))
    except Exception as e:
        
        st.error(f"Error: {e}")
#python -m streamlit run app.py 