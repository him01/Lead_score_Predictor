from fastapi import FastAPI, File, UploadFile, HTTPException
from pycaret.classification import load_model, predict_model
from fastapi.responses import Response
import pandas as pd
import io

app = FastAPI()

# Load the trained model
model = load_model('best_model')

# Store predictions in a dictionary (in-memory) for demonstration purposes
predictions_storage = {}

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Lead Scoring API!"}

@app.post("/upload-predict")
async def upload_predict(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    data = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Predict lead scores
    predictions = predict_model(model, data=data, raw_score=True)

    # Categorize leads
    predictions['Lead Category'] = pd.cut(predictions['prediction_score_1'] * 100, bins=[0, 33, 66, 100], labels=['Cold Lead', 'Warm Lead', 'Hot Lead'])

    # Store the predictions in memory (using prospect_id as key)
    for _, row in predictions.iterrows():
        predictions_storage[row['Prospect ID']] = row.to_dict()

    return predictions.to_dict(orient='records')

@app.get("/lead-details/{prospect_id}")
async def lead_details(prospect_id: str):
    # Retrieve specific lead details
    lead = predictions_storage.get(prospect_id)
    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    return lead

@app.get("/download-predictions")
async def download_predictions():
    # Convert stored predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions_storage.values())
    
    # Convert predictions to CSV
    csv = predictions_df.to_csv(index=False)
    
    return Response(content=csv, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})
