from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "AImonEar"
    API_V1_STR: str = "/api/v1"

    # Database (PostgreSQL)
    DATABASE_URL: str

    # Security (JWT & Hash)
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    ALGORITHM: str = "HS256"

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()