import numpy as np
import os
import json


class AImonEarNetwork:
    def __init__(self):
        # Carrega pesos se existirem, senão inicializa aleatório (simplificado)
        # Em produção, você chamaria um método 'load_weights'
        self.weights = np.random.rand(12, 24) - 0.5
        self.bias = np.random.rand(1, 24) - 0.5

        # Simulação de labels (Gere isso dinamicamente no futuro)
        self.labels = [f"Chord_{i}" for i in range(24)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, input_vector: list[int]):
        # Forward Pass
        x = np.array(input_vector).reshape(1, -1)
        z = np.dot(x, self.weights) + self.bias
        prob = self.sigmoid(z)[0]

        idx = np.argmax(prob)
        return self.labels[idx], float(prob[idx])


# Instância global para ser importada
ai_brain = AImonEarNetwork()