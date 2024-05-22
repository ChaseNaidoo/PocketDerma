from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from passlib.context import CryptContext

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
DATABASE_URL = "sqlite:///./users.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)

def create_user(db, username: str, email: str, password: str):
    hashed_password = pwd_context.hash(password)
    db_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_email(db, email: str):
    return db.query(User).filter(User.email == email).first()

def get_user_by_username(db, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db, email: str, password: str):
    user = get_user_by_email(db, email)
    if user and user.verify_password(password):
        return user
    return False

Base.metadata.create_all(bind=engine)
