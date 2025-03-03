import logging
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class QuantumCircuitVisualizer:
    def __init__(self):
        """Initialize the quantum circuit visualizer"""
        self.logger = logging.getLogger(__name__)
        self._circuit_data = None
        self._complexity_metrics = None

    def analyze_circuit_complexity(self, circuit) -> Dict:
        """Analyze the complexity of a quantum circuit with error handling"""
        try:
            # Calculate basic metrics
            metrics = {
                'depth': circuit.depth() if hasattr(circuit, 'depth') else 0,
                'width': circuit.width() if hasattr(circuit, 'width') else 0,
                'size': circuit.size() if hasattr(circuit, 'size') else 0,
                'num_qubits': circuit.num_qubits if hasattr(circuit, 'num_qubits') else 0,
                'num_clbits': circuit.num_clbits if hasattr(circuit, 'num_clbits') else 0,
                'gate_counts': self._count_gate_types(circuit),
                'layers': self._analyze_circuit_layers(circuit)
            }

            # Calculate advanced metrics with error handling
            metrics.update({
                'quantum_volume': self._estimate_quantum_volume(metrics),
                'parallelism_score': self._calculate_parallelism(metrics),
                'gate_depth_distribution': self._analyze_gate_depth_distribution(circuit)
            })

            self._complexity_metrics = metrics
            return metrics

        except Exception as e:
            self.logger.error(f"Error analyzing circuit complexity: {str(e)}")
            return self._get_default_metrics()

    def _count_gate_types(self, circuit) -> Dict[str, int]:
        """Count different types of gates in the circuit with error handling"""
        try:
            gate_counts = {}
            if hasattr(circuit, 'data'):
                for instruction in circuit.data:
                    gate_name = instruction[0].name
                    gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            return gate_counts
        except Exception as e:
            self.logger.error(f"Error counting gates: {str(e)}")
            return {}

    def _analyze_circuit_layers(self, circuit) -> List[Dict]:
        """Analyze the circuit layer by layer with error handling"""
        try:
            layers = []
            current_layer = set()
            layer_index = 0

            if hasattr(circuit, 'data'):
                for instruction in circuit.data:
                    # Get qubit indices safely
                    qubits = []
                    for qubit in instruction[1]:
                        # Try different ways to get qubit index
                        if hasattr(qubit, 'index'):
                            qubits.append(qubit.index)
                        elif hasattr(qubit, '_index'):
                            qubits.append(qubit._index)
                        else:
                            # Fallback to position in register
                            reg = getattr(qubit, 'register', None)
                            if reg:
                                qubits.append(reg.index(qubit))
                            else:
                                self.logger.warning(f"Could not determine index for qubit: {qubit}")
                                continue

                    # Check if gate can be added to current layer
                    if any(qubit in current_layer for qubit in qubits):
                        layers.append({
                            'index': layer_index,
                            'gates': list(current_layer)
                        })
                        current_layer = set()
                        layer_index += 1

                    # Add gate to current layer
                    for qubit in qubits:
                        current_layer.add(qubit)

            # Add final layer if not empty
            if current_layer:
                layers.append({
                    'index': layer_index,
                    'gates': list(current_layer)
                })

            return layers
        except Exception as e:
            self.logger.error(f"Error analyzing layers: {str(e)}")
            return []

    def _estimate_quantum_volume(self, metrics: Dict) -> float:
        """Estimate the quantum volume of the circuit"""
        try:
            depth = metrics['depth']
            width = metrics['num_qubits']
            # Simple estimation based on depth and width
            return 2 ** min(depth, width)
        except Exception as e:
            self.logger.error(f"Error estimating quantum volume: {str(e)}")
            return 1.0

    def _calculate_parallelism(self, metrics: Dict) -> float:
        """Calculate the parallelism score of the circuit"""
        try:
            total_gates = metrics['size']
            depth = metrics['depth']
            if depth == 0:
                return 0.0
            return total_gates / depth
        except Exception as e:
            self.logger.error(f"Error calculating parallelism: {str(e)}")
            return 0.0

    def _analyze_gate_depth_distribution(self, circuit) -> Dict[int, int]:
        """Analyze the distribution of gates across circuit depth"""
        try:
            distribution = {}
            current_depth = 0

            if hasattr(circuit, 'data'):
                for instruction in circuit.data:
                    distribution[current_depth] = distribution.get(current_depth, 0) + 1
                    if instruction[0].name not in ['barrier', 'snapshot']:
                        current_depth += 1

            return distribution
        except Exception as e:
            self.logger.error(f"Error analyzing gate distribution: {str(e)}")
            return {0: 0}

    def get_visualization_data(self) -> Dict:
        """Get the visualization data for the circuit"""
        if not self._complexity_metrics:
            return self._get_default_visualization()

        return {
            'metrics': self._complexity_metrics,
            'visualization': {
                'layers': self._prepare_layer_visualization(),
                'gate_distribution': self._prepare_gate_distribution(),
                'complexity_scores': self._calculate_complexity_scores()
            }
        }

    def _prepare_layer_visualization(self) -> List[Dict]:
        """Prepare layer-wise visualization data"""
        if not self._complexity_metrics:
            return []

        return [
            {
                'layer_index': layer['index'],
                'gate_count': len(layer['gates']),
                'qubit_usage': layer['gates']
            }
            for layer in self._complexity_metrics['layers']
        ]

    def _prepare_gate_distribution(self) -> Dict:
        """Prepare gate distribution visualization data"""
        if not self._complexity_metrics:
            return {'gate_counts': {}, 'depth_distribution': {}}

        return {
            'gate_counts': self._complexity_metrics['gate_counts'],
            'depth_distribution': self._complexity_metrics['gate_depth_distribution']
        }

    def _calculate_complexity_scores(self) -> Dict:
        """Calculate normalized complexity scores"""
        if not self._complexity_metrics:
            return self._get_default_scores()

        max_depth = 50  # Reference maximum depth
        max_gates = 100  # Reference maximum gates

        depth_score = min(1.0, self._complexity_metrics['depth'] / max_depth)
        size_score = min(1.0, self._complexity_metrics['size'] / max_gates)

        return {
            'depth_complexity': depth_score,
            'size_complexity': size_score,
            'quantum_volume_normalized': min(1.0, np.log2(self._complexity_metrics['quantum_volume']) / 10),
            'parallelism_efficiency': min(1.0, self._complexity_metrics['parallelism_score'] / 2)
        }

    def _get_default_metrics(self) -> Dict:
        """Return default metrics when analysis fails"""
        return {
            'depth': 0,
            'width': 0,
            'size': 0,
            'num_qubits': 0,
            'num_clbits': 0,
            'gate_counts': {},
            'layers': [],
            'quantum_volume': 1,
            'parallelism_score': 0,
            'gate_depth_distribution': {0: 0}
        }

    def _get_default_scores(self) -> Dict:
        """Return default complexity scores"""
        return {
            'depth_complexity': 0,
            'size_complexity': 0,
            'quantum_volume_normalized': 0,
            'parallelism_efficiency': 0
        }

    def _get_default_visualization(self) -> Dict:
        """Return default visualization data"""
        return {
            'metrics': self._get_default_metrics(),
            'visualization': {
                'layers': [],
                'gate_distribution': {
                    'gate_counts': {},
                    'depth_distribution': {}
                },
                'complexity_scores': self._get_default_scores()
            }
        }

def get_circuit_visualization(circuit) -> Dict:
    """Get visualization data for a quantum circuit with error handling"""
    try:
        visualizer = QuantumCircuitVisualizer()
        metrics = visualizer.analyze_circuit_complexity(circuit)
        return visualizer.get_visualization_data()
    except Exception as e:
        logger.error(f"Error in get_circuit_visualization: {str(e)}")
        return QuantumCircuitVisualizer()._get_default_visualization()