from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd
from src.prediction import ModelPredictor
app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        else:
            return JSONResponse(content={"error": "Input payload must be a JSON object or list of objects."}, status_code=400)
        predictor = ModelPredictor()
        preprocessed_data = predictor.preprocess_input(data)
        predictions = predictor.predict(preprocessed_data)
        return JSONResponse(content={"predictions": predictions.tolist()})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)