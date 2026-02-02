from fastapi import FastAPI
from app.core.config import settings
from app.api import auth, users, music # Assumindo que você criará users.py similar ao auth.py

app = FastAPI(title=settings.PROJECT_NAME)

# Incluir Rotas
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(music.router, prefix=f"{settings.API_V1_STR}/music", tags=["music"])

# (Opcional) Rota raiz para teste
@app.get("/")
def root():
    return {"message": "Welcome to AImonEar API"}