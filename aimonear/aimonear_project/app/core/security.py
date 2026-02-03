import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Any, Union
from app.core.config import settings

def create_access_token(subject: Union[str, Any]) -> str:
    """Gera um Token JWT com validade definida."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"exp": expire, "sub": str(subject)}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica se a senha bate com o hash (Bcrypt)."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    """Gera o hash seguro para salvar no banco."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')