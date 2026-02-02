from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.database.connection import get_db
from app.database.models import User, TrainingSession
from app.schemas.music_schema import NoteInput, PredictionResponse
from app.services.music_theory import get_note_vector
from app.ml.engine import ai_brain

router = APIRouter()


@router.post("/identify", response_model=PredictionResponse)
def identify_chord(
        input_data: NoteInput,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    # 1. Converte notas em vetor
    vector = get_note_vector(input_data.notes)

    # 2. IA processa
    chord_name, confidence = ai_brain.predict(vector)

    # 3. Salva no histórico
    session_db = TrainingSession(
        user_id=current_user.id,
        input_notes=",".join(input_data.notes),
        predicted_chord=chord_name,
        confidence=confidence
    )
    db.add(session_db)
    db.commit()

    return {
        "chord_name": chord_name,
        "confidence": confidence,
        "notes_played": input_data.notes
    }