import numpy as np

NOTES_ORDER = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def get_note_vector(notes: list[str]) -> list[int]:
    """Converte lista ['C', 'E', 'G'] em vetor One-Hot [1, 0, 0, 0, 1...]"""
    vector = [0] * 12
    for note in notes:
        # Simplificação: Remove números de oitava se houver (ex: C4 -> C)
        clean_note = ''.join([i for i in note if not i.isdigit()])
        if clean_note in NOTES_ORDER:
            idx = NOTES_ORDER.index(clean_note)
            vector[idx] = 1
    return vector