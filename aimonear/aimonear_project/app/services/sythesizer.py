import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional
import httpx
import numpy as np
from scipy import signal

from app.ml.engine import ai_brain
from app.services.music_theory import NOTES_ORDER

# Frequencias fundamentais (Hz) da 4a oitava (A4 = 440 Hz)
NOTE_FREQUENCIES_HZ: Dict[str, float] = {
    "C": 261.63,
    "C#": 277.18,
    "D": 293.66,
    "D#": 311.13,
    "E": 329.63,
    "F": 349.23,
    "F#": 369.99,
    "G": 392.00,
    "G#": 415.30,
    "A": 440.00,
    "A#": 466.16,
    "B": 493.88,
}


def _clean_note_name(note: str) -> str:
    """Remove digitos de oitava (ex: 'C4' -> 'C') mantendo sustenidos."""
    return "".join([ch for ch in note.strip().upper() if not ch.isdigit()])


def synthesize_chord_audio(
    notes: List[str],
    duration_sec: float = 0.32,
    sample_rate: int = 48000,
    frame_size: int = 128,
    detune_cents: float = 0.0,
) -> Dict[str, Any]:
    """
    Sintese Aditiva Polifonica em Tempo Real para Treinamento Auditivo:
    Gera a forma de onda de alta fidelidade (48 kHz) das notas do acorde com
    harmonicos naturais (1f, 2f, 3f, 4f) e envelope ADSR, segmentada em blocos
    de 128 frames compativeis com a ponte Zero-Copy / AudioWorklet (WebAssembly).
    """
    t0 = time.perf_counter()
    total_samples = int( max(frame_size * 4, round(duration_sec * sample_rate)) )
    # Ajusta para multiplo exato de 128 frames (bloco Wasm/SonoBus)
    num_frames = max(4, total_samples // frame_size)
    total_samples = num_frames * frame_size

    t = np.linspace(0.0, total_samples / sample_rate, total_samples, endpoint=False)
    waveform = np.zeros(total_samples, dtype=float)

    # Fator de desvio de micro-afinacao (100 cents = 1 semitom)
    detune_ratio = 2.0 ** (detune_cents / 1200.0)
    active_frequencies: List[float] = []

    for raw_note in notes:
        clean = _clean_note_name(raw_note)
        base_freq = NOTE_FREQUENCIES_HZ.get(clean)
        if base_freq is None:
            continue
        freq = base_freq * detune_ratio
        active_frequencies.append(round(freq, 2))

        # Sintese aditiva: fundamental + 3 harmonicos superiores
        note_wave = (
            1.00 * np.sin(2.0 * np.pi * freq * t)
            + 0.45 * np.sin(2.0 * np.pi * (2.0 * freq) * t)
            + 0.20 * np.sin(2.0 * np.pi * (3.0 * freq) * t)
            + 0.08 * np.sin(2.0 * np.pi * (4.0 * freq) * t)
        )
        waveform += note_wave

    if len(active_frequencies) > 0:
        waveform /= len(active_frequencies)

    # Envelope ADSR suave para preservar transientes naturais de ataque
    attack_len = max(1, int(0.08 * total_samples))
    decay_len = max(1, int(0.15 * total_samples))
    release_len = max(1, int(0.20 * total_samples))
    sustain_len = max(1, total_samples - (attack_len + decay_len + release_len))

    envelope = np.concatenate([
        np.linspace(0.0, 1.0, attack_len, endpoint=False),
        np.linspace(1.0, 0.78, decay_len, endpoint=False),
        np.full(sustain_len, 0.78),
        np.linspace(0.78, 0.0, total_samples - (attack_len + decay_len + sustain_len)),
    ])
    waveform = waveform * envelope

    # Normalizacao segura em float32 (-1.0 a 1.0)
    peak = float(np.max(np.abs(waveform)))
    if peak > 1e-8:
        waveform = 0.92 * (waveform / peak)

    frames = waveform.reshape(num_frames, frame_size)
    rms_energy = float(np.sqrt(np.mean(waveform ** 2)))
    synthesis_time_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "waveform": waveform,
        "frames": frames,
        "sample_rate": sample_rate,
        "frame_size": frame_size,
        "num_frames": num_frames,
        "active_frequencies_hz": active_frequencies,
        "rms_energy": round(rms_energy, 4),
        "synthesis_time_ms": round(synthesis_time_ms, 4),
    }


def process_jitter_buffer_plc(
    frames: np.ndarray,
    network_jitter_ms: float = 4.2,
    one_way_network_delay_ms: float = 14.5,
    packet_loss_rate: float = 0.05,
) -> Dict[str, Any]:
    """
    Simula a recepcao P2P via Audio over OSC (AOO / SonoBus Wasm) sob Jitter de rede
    domestica e aciona o Jitter Buffer Neural (AImonEar Edge AI) para Ocultacao de
    Perda de Pacotes (PLC), verificando o cumprimento do Efeito Haas e do Limiar de
    Performance de Conjunto (EPT < 30 ms).
    """
    num_frames, frame_size = frames.shape
    lost_Frames = max(1, int(round(num_frames * packet_loss_rate))) if packet_loss_rate > 0 else 0

    history_frames = frames[: max(4, num_frames - lost_Frames)]
    plc_result = ai_brain.conceal_packet_loss(
        history_frames=history_frames,
        lost_frames_count=max(1, lost_Frames),
    )

    # Latencia interna do bloco Wasm (128 samples @ 48kHz = 2.66ms buffer + processamento neural)
    dsp_wasm_processing_ms = float(plc_result["plc_inference_time_ms"])
    total_estimated_latency_ms = round(
        one_way_network_delay_ms + network_jitter_ms + dsp_wasm_processing_ms, 3
    )

    # Avaliacao psicoacustica segundo Chris Chafe (CCRMA/Stanford) e Helmut Haas
    haas_fusion_preserved = total_estimated_latency_ms < 30.0
    if total_estimated_latency_ms < 10.0:
        rhythmic_tendency = "slight_rush_anticipation"
    elif total_estimated_latency_ms <= 25.0:
        rhythmic_tendency = "optimal_chamber_sync"
    elif total_estimated_latency_ms <= 30.0:
        rhythmic_tendency = "borderline_ept_sync"
    else:
        rhythmic_tendency = "haas_breakdown_drag_risk"

    return {
        "total_latency_ms": total_estimated_latency_ms,
        "network_delay_ms": round(one_way_network_delay_ms, 3),
        "network_jitter_ms": round(network_jitter_ms, 3),
        "edge_ai_plc_inference_ms": dsp_wasm_processing_ms,
        "within_haas_30ms_threshold": bool(haas_fusion_preserved),
        "within_wasm_1_5ms_budget": bool(plc_result["within_wasm_1_5ms_budget"]),
        "packets_concealed": int(plc_result["lost_frames_recovered"]),
        "spectral_continuity_score": float(plc_result["spectral_continuity_score"]),
        "psychoacoustic_sync_state": rhythmic_tendency,
    }


def anonymize_user_identifier(user_id: int, email: str = "") -> str:
    """
    Gera identificador pseudonimizado com HMAC-SHA256 + pepper em conformidade
    com a LGPD (compativel com app/core/security.py do repositorio recommender).
    """
    pepper = os.getenv("LGPD_PEPPER_SECRET", "master-musica-lgpd-pepper-2026")
    raw_value = f"user:{user_id}:{email.lower().strip()}"
    return hmac.new(
        pepper.encode("utf-8"),
        raw_value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def extract_biometric_telemetry(
    user_id: int,
    notes: List[str],
    chord_confidence: float,
    synth_data: Dict[str, Any],
    plc_metrics: Dict[str, Any],
    user_email: str = "",
    pitch_deviation_cents: float = 4.5,
    rhythmic_jitter_ms: float = 6.0,
    session_duration_minutes: float = 18.0,
) -> Dict[str, Any]:
    """
    Sensoriamento Adaptativo Passivo (Edge AI - AImonEar):
    Extrai metadados enxutos da performance (micro-afinacao, estabilidade ritmica,
    fadiga biomecanica e risco de frustracao cognitiva) sem transmitir audio bruto
    para a nuvem, em estrito cumprimento da LGPD.
    """
    waveform: np.ndarray = synth_data["waveform"]
    sample_rate: int = synth_data["sample_rate"]

    # Calcula centroide espectral via Periodograma de Welch (SciPy) para aferir brilho/ataque
    freqs, psd = signal.welch(waveform, fs=sample_rate, nperseg=min(512, len(waveform)))
    psd_sum = float(np.sum(psd)) + 1e-12
    spectral_centroid_hz = float(np.sum(freqs * psd) / psd_sum)

    # Indice de estabilidade de afinacao (0.0 a 1.0, onde desvios > 50 cents degradam)
    pitch_stability_score = float(np.clip(1.0 - (abs(pitch_deviation_cents) / 50.0), 0.0, 1.0))

    # Indice de precisao micro-ritmica (baseado no limiar EPT de 30 ms)
    rhythmic_precision_score = float(np.clip(1.0 - (rhythmic_jitter_ms / 30.0), 0.0, 1.0))

    # Calculo do indice de fadiga motora/cognitiva e frustracao aguda (0.0 a 1.0)
    duration_factor = float(np.clip(session_duration_minutes / 60.0, 0.0, 1.0))
    fatigue_index = float(
        np.clip(
            0.35 * (1.0 - pitch_stability_score)
            + 0.35 * (1.0 - rhythmic_precision_score)
            + 0.20 * duration_factor
            + 0.10 * (1.0 - chord_confidence),
            0.0,
            1.0,
        )
    )

    if fatigue_index >= 0.65:
        pedagogical_alert = "high_fatigue_reduce_complexity"
        recommended_action = "Sugerir pausa ativa ou revisao leve de repertorio consolidado"
    elif fatigue_index >= 0.40:
        pedagogical_alert = "moderate_load_stabilize_tempo"
        recommended_action = "Manter andamento moderado com exercicios de percepcao intervalar"
    else:
        pedagogical_alert = "optimal_flow_advance_curriculum"
        recommended_action = "Avancar na trilha pedagogica com novos encadeamentos harmonicos"

    return {
        "anonymized_student_hash": anonymize_user_identifier(user_id=user_id, email=user_email),
        "notes_count": len(notes),
        "spectral_centroid_hz": round(spectral_centroid_hz, 2),
        "pitch_deviation_cents": round(float(pitch_deviation_cents), 2),
        "pitch_stability_score": round(pitch_stability_score, 4),
        "rhythmic_precision_score": round(rhythmic_precision_score, 4),
        "fatigue_index": round(fatigue_index, 4),
        "pedagogical_alert": pedagogical_alert,
        "recommended_pedagogical_action": recommended_action,
        "plc_network_telemetry": plc_metrics,
        "lgpd_raw_audio_transmitted": False,
    }


def send_telemetry_to_recommender(
    user_id: int,
    predicted_chord: str,
    confidence: float,
    telemetry: Dict[str, Any],
    recommender_base_url: Optional[str] = None,
    timeout_sec: float = 2.5,
) -> Dict[str, Any]:
    """
    Envia os metadados enxutos de telemetria biometrica do AImonEar (Edge AI) para o
    Orquestrador Cloud Recommender via httpx, permitindo o ajuste dinamico das trilhas
    do LMS e o calculo de pesos hibridos (Grafo + RandomForest).
    Opera em modo resiliente (non-blocking fallback) caso o servico Recommender esteja offline.
    """
    base_url = (
        recommender_base_url
        or os.getenv("RECOMMENDER_API_URL", "http://localhost:8001")
    ).rstrip("/")

    payload = {
        "source": "aimonear_edge_ai",
        "student_hash": telemetry["anonymized_student_hash"],
        "user_id": user_id,
        "predicted_chord": predicted_chord,
        "chord_confidence": confidence,
        "fatigue_index": telemetry["fatigue_index"],
        "pitch_stability_score": telemetry["pitch_stability_score"],
        "rhythmic_precision_score": telemetry["rhythmic_precision_score"],
        "pedagogical_alert": telemetry["pedagogical_alert"],
        "recommended_pedagogical_action": telemetry["recommended_pedagogical_action"],
        "total_latency_ms": telemetry["plc_network_telemetry"]["total_latency_ms"],
        "within_haas_30ms_threshold": telemetry["plc_network_telemetry"]["within_haas_30ms_threshold"],
    }

    # Calcula peso de interacao sugerido para o grafo do Recommender
    interaction_weight = round(
        float(
            0.50 * confidence
            + 0.25 * telemetry["pitch_stability_score"]
            + 0.25 * telemetry["rhythmic_precision_score"]
        ),
        4,
    )
    payload["suggested_graph_edge_weight"] = interaction_weight

    try:
        with httpx.Client(timeout=timeout_sec) as client:
            response = client.post(
                f"{base_url}/recommendations/telemetry",
                json=payload,
                params={"user_id": user_id},
            )
            if response.status_code < 400:
                return {
                    "synced_with_recommender": True,
                    "recommender_url": f"{base_url}/recommendations/telemetry",
                    "status_code": response.status_code,
                    "suggested_graph_edge_weight": interaction_weight,
                    "recommender_response": response.json(),
                }
            # Fallback para a rota GET /recommendations existente no repositorio recommender
            fallback_resp = client.get(
                f"{base_url}/recommendations",
                params={"user_id": user_id, "top_k": 3},
            )
            return {
                "synced_with_recommender": fallback_resp.status_code < 400,
                "recommender_url": f"{base_url}/recommendations",
                "status_code": fallback_resp.status_code,
                "suggested_graph_edge_weight": interaction_weight,
                "recommender_response": (
                    fallback_resp.json() if fallback_resp.status_code < 400 else None
                ),
            }
    except httpx.HTTPError as exc:
        return {
            "synced_with_recommender": False,
            "recommender_url": base_url,
            "status": "queued_shadow_mode",
            "reason": f"Recommender service unreachable ({exc.__class__.__name__})",
            "suggested_graph_edge_weight": interaction_weight,
            "telemetry_payload_ready": payload,
        }
