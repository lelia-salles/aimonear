from app.core.security import verify_password, create_access_token
from app.database.models import User
from app.schemas.user_schema import Token
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from aimonear.aimonear.aimonear_project.app.database.connection import get_db

router = APIRouter()


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    token = create_access_token(subject=user.id)
    return {"access_token": token, "token_type": "bearer"}