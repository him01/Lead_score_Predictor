# Lead_score_Predictor
This project is a FastAPI-based web application that predicts lead conversion scores based on input data. The model used is a RandomForestClassifier trained on historical lead data.

Features
Lead Scoring Prediction: Upload a CSV file with lead data, and get predictions on whether the leads will convert.

Endpoints:
/upload-predict: Upload a CSV file to get lead conversion predictions.

Getting Started
# Prerequisites
Make sure you have the following installed:
Python 3.8+
FastAPI
Uvicorn
Pandas
Scikit-learn
PyCaret (if used)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/lead-scoring-api.git
cd lead-scoring-api
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the dependencies:

bash
Copy code
pip install -r requirements.txt
Ensure the trained model file (Random_forest_model_Leadscore.pkl) is in the project directory.

Running the Application
Run the FastAPI application using Uvicorn:

bash
Copy code
uvicorn app:app --reload
The API will be available at http://127.0.0.1:8000.

Usage
Access the API Documentation:
Navigate to http://127.0.0.1:8000/docs to explore the available endpoints using the interactive Swagger UI.

# Predict Lead Conversion:

Use the /upload-predict endpoint to upload a CSV file with lead data and receive predictions.

Example request:

bash
Copy code
curl -X POST "http://127.0.0.1:8000/upload-predict" -F "file=@sample_leads_data.csv"
The response will contain predictions indicating which leads are likely to convert.

# Project Structure
app.py: The FastAPI application that defines the routes and logic for prediction.
best_model.pkl: Saved model used for predictions.

License
This project is licensed under the MIT License.

Contact
For any inquiries, please reach out to your himanshusharma0067@gmail.com
