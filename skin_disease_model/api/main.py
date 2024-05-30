#!/usr/bin/env python3

# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import schemas
from utils import get_hashed_password, create_access_token, create_refresh_token, settings
from auth_bearer import JWTBearer
from models import User
from utils import verify_password
import jwt
from datetime import datetime
from functools import wraps
from jwt import PyJWTError
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get JWT secret key from environment variables
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
JWT_REFRESH_SECRET_KEY = os.getenv('JWT_REFRESH_SECRET_KEY')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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
async def predict(file: UploadFile = File(...)):
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

# Function to create a session for database operations
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Define the user registration endpoint
@app.post("/signup")
def register_user(user: schemas.UserCreate, session: Session = Depends(get_session)):
    existing_user = session.query(models.User).filter_by(email=user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    encrypted_password = get_hashed_password(user.password)

    new_user = models.User(username=user.username, email=user.email, password=encrypted_password)

    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    return {"message": "User created successfully"}

@app.post('/login', response_model=schemas.TokenSchema)
def login(request: schemas.requestdetails, db: Session = Depends(get_session)):
    user = db.query(User).filter(User.email == request.email).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect email")
    hashed_pass = user.password
    if not verify_password(request.password, hashed_pass):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password"
        )
    
    access=create_access_token(user.id)
    refresh = create_refresh_token(user.id)

    token_db = models.TokenTable(user_id=user.id,  access_token=access,  refresh_token=refresh, status=True)
    db.add(token_db)
    db.commit()
    db.refresh(token_db)
    return {
        "access_token": access,
        "refresh_token": refresh,
    }

@app.get('/getusers')
def getusers( dependencies=Depends(JWTBearer()),session: Session = Depends(get_session)):
    user = session.query(models.User).all()
    return user

@app.post('/change-password')
def change_password(request: schemas.changepassword, db: Session = Depends(get_session)):
    user = db.query(models.User).filter(models.User.email == request.email).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User not found")
    
    if not verify_password(request.old_password, user.password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid old password")
    
    encrypted_password = get_hashed_password(request.new_password)
    user.password = encrypted_password
    db.commit()
    
    return {"message": "Password changed successfully"}

@app.post('/logout')
def logout(dependencies=Depends(JWTBearer()), db: Session = Depends(get_session)):
    token=dependencies
    payload = jwt.decode(token, JWT_SECRET_KEY, ALGORITHM)
    user_id = payload['sub']
    token_record = db.query(models.TokenTable).all()
    info=[]
    for record in token_record :
        print("record",record)
        if (datetime.utcnow() - record.created_date).days >1:
            info.append(record.user_id)
    if info:
        existing_token = db.query(models.TokenTable).where(TokenTable.user_id.in_(info)).delete()
        db.commit()
        
    existing_token = db.query(models.TokenTable).filter(models.TokenTable.user_id == user_id, models.TokenTable.access_token==token).first()
    if existing_token:
        existing_token.status=False
        db.add(existing_token)
        db.commit()
        db.refresh(existing_token)
    return {"message":"Logout Successfully"} 

def token_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
    
        payload = jwt.decode(kwargs['dependencies'], JWT_SECRET_KEY, ALGORITHM)
        user_id = payload['sub']
        data= kwargs['session'].query(models.TokenTable).filter_by(user_id=user_id, access_token=kwargs['dependencies'], status=True).first()
        if data:
            return func(kwargs['dependencies'],kwargs['session'])
        else:
            return {'msg': "Token blocked"}
        
    return wrapper

models.Base.metadata.create_all(bind=engine)

# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
