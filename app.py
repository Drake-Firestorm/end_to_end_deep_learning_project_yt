import os
import subprocess

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.end_to_end_deep_learning_project_yt.pipeline.stage_05_prediction import (
    PredictionPipeline,
)

MODEL_PATH = "artifacts/training/model.keras"

app = FastAPI(title="ML Classifier App")


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

    def get_predictions(self):
        self.classifier = PredictionPipeline(self.filename)
        return self.classifier.predict()


clApp = ClientApp()

ALLOWED_TYPES = {"image/jpeg"}  # .jpg uploads


def _save_upload(file: UploadFile, dest: str) -> None:
    with open(dest, "wb") as f:
        f.write(file.file.read())


@app.get("/")
def home():
    return {"message": "home"}


@app.get("/train")
def train():
    try:
        subprocess.run([".venv/bin/python", "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e.returncode}")
    return {"message": "Training done sucessfully!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=409, detail="Model not trained yet. Call GET /train first."
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only .jpg files are accepted")

    try:
        _save_upload(file, clApp.filename)  # saves to inputImage.jpg
        result = clApp.get_predictions()
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
