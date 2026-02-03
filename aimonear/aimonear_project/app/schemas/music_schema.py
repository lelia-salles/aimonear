from pydantic import BaseModel
from typing import List

class NoteInput(BaseModel):
    notes: List[str] # Ex: ["C", "E", "G"]

class PredictionResponse(BaseModel):
    chord_name: str  # Ex: "C Major"
    confidence: float
    notes_played: List[str]