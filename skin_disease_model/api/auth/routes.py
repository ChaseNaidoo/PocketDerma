from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from auth.models import SessionLocal, create_user, authenticate_user, get_user_by_email

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/signup")
async def signup(username: str, email: str, password: str, db: Session = Depends(get_db)):
    user = get_user_by_email(db, email)
    if user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    create_user(db, username, email, password)
    return {"message": "User created successfully"}

@router.post("/login")
async def login(email: str, password: str, db: Session = Depends(get_db)):
    user = authenticate_user(db, email, password)
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid credentials")
    return {"message": "Login successful"}
