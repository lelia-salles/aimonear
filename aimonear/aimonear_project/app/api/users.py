from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Annotated

# Project imports
from app.database.connection import get_db
from app.database.models import User
from app.schemas.user_schema import UserCreate, UserResponse
from app.core.security import get_password_hash
from app.api.deps import get_current_user

router = APIRouter()


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Create system user (Sign Up).
    1. Verify if email already exists.
    2. Passkey criptography (Hash).
    3. Save data on database.
    """
    # 1. Verify duplicity
    user_exists = db.query(User).filter(User.email == user.email).first()
    if user_exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # 2. Prepare object (passkey hash)
    hashed_password = get_password_hash(user.password)

    new_user = User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        is_active=True
    )

    # 3. Save to Postgres
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)  # Recarrega para pegar o ID gerado
        return new_user
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    """
    Returns user data whe logged in.
    Demands header: 'Authorization: Bearer <token>'
    """
    return current_user