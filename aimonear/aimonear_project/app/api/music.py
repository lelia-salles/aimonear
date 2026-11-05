from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.database.connection import get_db
from app.database.models import User, TrainingSession
from app.schemas.music_schema import NoteInput, PredictionResponse
from app.services.music_theory import get_note_vector
from app.services.sythesizer import (
    synthesize_chord_audio,
    process_jitter_buffer_plc,
    extract_biometric_telemetry,
    send_telemetry_to_recommender,
)
from app.ml.engine import ai_brain

router = APIRouter()


@router.post("/identify", response_model=PredictionResponse)
def identify_chord(
    input_data: NoteInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1. Converte notas em vetor cromatico de 12 posicoes
    vector = get_note_vector(input_data.notes)

    # 2. IA (AImonEar) identifica o acorde com pesos calibrados reais
    chord_name, confidence = ai_brain.predict(vector)

    # 3. Sintetiza o acorde em blocos de 128 frames (48 kHz) e executa Jitter Buffer Neural / PLC
    synth_data = synthesize_chord_audio(
        notes=input_data.notes,
        detune_cents=input_data.pitch_deviation_cents,
    )
    plc_metrics = process_jitter_buffer_plc(
        frames=synth_data["frames"],
        network_jitter_ms=input_data.network_jitter_ms,
        one_way_network_delay_ms=input_data.one_way_delay_ms,
        packet_loss_rate=input_data.packet_loss_rate,
    )

    # 4. Extrai telemetria biometrica passiva anonimizada (LGPD / HMAC-SHA256)
    telemetry = extract_biometric_telemetry(
        user_id=current_user.id,
        notes=input_data.notes,
        chord_confidence=confidence,
        synth_data=synth_data,
        plc_metrics=plc_metrics,
        user_email=getattr(current_user, "email", ""),
        pitch_deviation_cents=input_data.pitch_deviation_cents,
        rhythmic_jitter_ms=input_data.network_jitter_ms,
        session_duration_minutes=input_data.session_duration_minutes,
    )

    # 5. Envia metadados de telemetria para o Orquestrador Cloud Recommender via httpx
    recommender_sync = send_telemetry_to_recommender(
        user_id=current_user.id,
        predicted_chord=chord_name,
        confidence=confidence,
        telemetry=telemetry,
    )

    # 6. Salva no historico do PostgreSQL
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
        "notes_played": input_data.notes,
        "telemetry": telemetry,
        "recommender_sync": recommender_sync,
    }
