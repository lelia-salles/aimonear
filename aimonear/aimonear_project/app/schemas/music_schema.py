from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class NoteInput(BaseModel):
    notes: List[str] = Field(..., description="Lista de notas tocadas, ex: ['C', 'E', 'G']")
    pitch_deviation_cents: float = Field(default=3.5, description="Desvio medio de micro-afinacao em cents")
    network_jitter_ms: float = Field(default=4.2, description="Jitter instantaneo da rede P2P em ms")
    one_way_delay_ms: float = Field(default=14.5, description="Latencia unidirecional da rede em ms")
    packet_loss_rate: float = Field(default=0.05, description="Taxa de perda de pacotes UDP (0.0 a 1.0)")
    session_duration_minutes: float = Field(default=15.0, description="Duracao acumulada da sessao em minutos")


class PredictionResponse(BaseModel):
    chord_name: str  # Ex: "C Major"
    confidence: float
    notes_played: List[str]
    telemetry: Optional[Dict[str, Any]] = None
    recommender_sync: Optional[Dict[str, Any]] = None
