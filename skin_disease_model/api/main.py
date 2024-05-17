#!/usr/bin/env python3

# Import necessary libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Create a FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained TensorFlow/Keras model
MODEL = tf.keras.models.load_model("../models/1")

# Define class names for the model's predictions
CLASS_NAMES = ["Healthy", "Melanoma"]

# Define a route for a ping endpoint
@app.get("/ping")
async def ping():
    """Check if the API is running."""
    return "Status: OK"

# Function to read an uploaded file as an image
def read_file_as_image(data) -> np.ndarray:
    """Read an uploaded file as an image."""
    image = np.array(Image.open(BytesIO(data)))
    return image

# Define a route for the prediction endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    """Predict the class of a plant disease from an uploaded image."""
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # Make predictions using the loaded model
    predictions = MODEL.predict(img_batch)

    # Get the predicted class and its confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    # Return the predicted class and confidence
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)