import random
import math
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class QuantumLikeState:
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """Initialize a quantum-like state with classical probabilities"""
        self.alpha = alpha  # Probability amplitude for |0⟩
        self.beta = beta    # Probability amplitude for |1⟩
        self.measured = False
        self.measured_value = None
        
        # Validate probability amplitudes
        if not math.isclose(alpha**2 + beta**2, 1.0, rel_tol=1e-9):
            logger.warning("Invalid probability amplitudes. Normalizing...")
            norm = math.sqrt(alpha**2 + beta**2)
            self.alpha /= norm
            self.beta /= norm

    def measure(self) -> int:
        """Measure the quantum-like state"""
        if not self.measured:
            # Randomly decide based on probability amplitudes
            self.measured_value = 0 if random.random() < self.alpha**2 else 1
            self.measured = True
            logger.debug(f"Measured state: {self.measured_value}")
        return self.measured_value

    def reset(self):
        """Reset to superposition state"""
        self.measured = False
        self.measured_value = None
        logger.debug("State reset to superposition")

    def apply_X(self):
        """Apply X (NOT) gate"""
        self.alpha, self.beta = self.beta, self.alpha
        logger.debug("Applied X gate")

    def apply_H(self):
        """Apply Hadamard gate"""
        new_alpha = (self.alpha + self.beta) / math.sqrt(2)
        new_beta = (self.alpha - self.beta) / math.sqrt(2)
        self.alpha, self.beta = new_alpha, new_beta
        logger.debug("Applied H gate")

    def __str__(self) -> str:
        return f"|ψ⟩ = {self.alpha:.3f}|0⟩ + {self.beta:.3f}|1⟩"

class EntangledPair:
    def __init__(self):
        """Initialize an entangled pair of quantum-like states"""
        self.qubit_a = QuantumLikeState()
        # Create inversely correlated state
        self.qubit_b = QuantumLikeState(alpha=self.qubit_a.beta, beta=self.qubit_a.alpha)
        logger.info("Created new entangled pair")

    def measure_pair(self) -> Tuple[int, int]:
        """Measure both qubits of the entangled pair"""
        a = self.qubit_a.measure()
        b = 1 - a  # Ensure perfect anti-correlation
        self.qubit_b.measured_value = b
        self.qubit_b.measured = True
        logger.debug(f"Measured entangled pair: A={a}, B={b}")
        return a, b

    def apply_X_to_pair(self):
        """Apply X gate to both qubits"""
        self.qubit_a.apply_X()
        self.qubit_b.apply_X()
        logger.debug("Applied X gate to entangled pair")

    def apply_H_to_pair(self):
        """Apply Hadamard gate to both qubits"""
        self.qubit_a.apply_H()
        self.qubit_b.apply_H()
        logger.debug("Applied H gate to entangled pair")

    def get_state_description(self) -> str:
        """Get a string description of the entangled state"""
        return f"Qubit A: {self.qubit_a}\nQubit B: {self.qubit_b}"

class ClassicalQuantumSimulator:
    def __init__(self):
        """Initialize the classical quantum simulator"""
        self.states = {}
        self.entangled_pairs = {}
        logger.info("Initialized Classical Quantum Simulator")

    def create_state(self, name: str, alpha: float = 0.5, beta: float = 0.5) -> QuantumLikeState:
        """Create and register a new quantum-like state"""
        state = QuantumLikeState(alpha, beta)
        self.states[name] = state
        logger.info(f"Created new state '{name}': {state}")
        return state

    def create_entangled_pair(self, name: str) -> EntangledPair:
        """Create and register a new entangled pair"""
        pair = EntangledPair()
        self.entangled_pairs[name] = pair
        logger.info(f"Created new entangled pair '{name}'")
        return pair

    def measure_state(self, name: str) -> Optional[int]:
        """Measure a registered quantum-like state"""
        state = self.states.get(name)
        if state:
            result = state.measure()
            logger.info(f"Measured state '{name}': {result}")
            return result
        logger.warning(f"State '{name}' not found")
        return None

    def measure_entangled_pair(self, name: str) -> Optional[Tuple[int, int]]:
        """Measure a registered entangled pair"""
        pair = self.entangled_pairs.get(name)
        if pair:
            result = pair.measure_pair()
            logger.info(f"Measured entangled pair '{name}': {result}")
            return result
        logger.warning(f"Entangled pair '{name}' not found")
        return None
