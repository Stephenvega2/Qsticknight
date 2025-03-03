import numpy as np
from typing import List, Tuple, Dict
import logging
from models import db, HeatMapData

logger = logging.getLogger(__name__)

class QuantumVisualizer:
    def __init__(self):
        """Initialize the quantum visualization with predefined positions and heat map configuration"""
        self.current_state = {
            'source': (100, 300),
            'entangled1': (300, 400),
            'entangled2': (300, 200),
            'target': (500, 300),
            'security_check': (400, 300)
        }
        self.colors = {
            'source': (0, 0, 1),      # Blue for source
            'entangled1': (1, 0, 0),  # Red for entangled state 1
            'entangled2': (1, 0, 0),  # Red for entangled state 2
            'target': (0, 1, 0),      # Green for target
            'security_check': (1, 0.5, 0)  # Orange for security check
        }
        self.heatmap_resolution = 50  # Grid size for heat map
        self.heatmap_data = np.zeros((self.heatmap_resolution, self.heatmap_resolution))
        self.captured_states = []  # Track captured states
        self._frames = None
        logger.info("Quantum visualizer initialized with heat map support")

    def update_heatmap(self, x: float, y: float, intensity: float = 1.0, state_type: str = 'quantum_state'):
        """Update heat map data at given coordinates"""
        # Convert coordinates to grid indices
        grid_x = int((x / 600) * self.heatmap_resolution)
        grid_y = int((y / 400) * self.heatmap_resolution)

        # Ensure indices are within bounds
        grid_x = max(0, min(grid_x, self.heatmap_resolution - 1))
        grid_y = max(0, min(grid_y, self.heatmap_resolution - 1))

        # Update heat map with decay
        decay_factor = 0.95
        self.heatmap_data *= decay_factor
        self.heatmap_data[grid_y, grid_x] += intensity

        # Track captured state
        self.captured_states.append({
            'x': x,
            'y': y,
            'intensity': intensity,
            'state_type': state_type,
            'color': self.colors.get(state_type, (1, 1, 1))
        })

    def get_state_snapshot(self, progress: float) -> Dict:
        """Get the current state of the visualization based on progress (0-1)"""
        if progress < 0.2:  # Phase 1: Initial state preparation
            t = progress * 5
            source_pos = self._interpolate(
                self.current_state['source'],
                (200, 300),
                t
            )
            # Update heat map for source position
            self.update_heatmap(source_pos[0], source_pos[1], 1.0, 'source')

            return {
                'positions': {
                    'source': source_pos,
                    'entangled1': self.current_state['entangled1'],
                    'entangled2': self.current_state['entangled2'],
                    'target': self.current_state['target'],
                    'security_check': self.current_state['security_check']
                },
                'colors': self.colors.copy(),
                'heatmap': self.heatmap_data.tolist(),
                'captured_states': self.captured_states
            }
        elif progress < 0.4:  # Phase 2: Entanglement generation
            t = (progress - 0.2) * 5
            pulse = abs(np.sin(t * np.pi * 2))
            colors = self.colors.copy()
            colors['entangled1'] = (1, pulse, pulse)
            colors['entangled2'] = (1, pulse, pulse)
            self.update_heatmap(self.current_state['entangled1'][0], self.current_state['entangled1'][1], pulse, 'entangled1')
            self.update_heatmap(self.current_state['entangled2'][0], self.current_state['entangled2'][1], pulse, 'entangled2')
            return {
                'positions': self.current_state.copy(),
                'colors': colors,
                'heatmap': self.heatmap_data.tolist(),
                'captured_states': self.captured_states
            }
        elif progress < 0.6:  # Phase 3: Security check
            t = (progress - 0.4) * 5
            colors = self.colors.copy()
            colors['security_check'] = (1, 0.5 + 0.5 * np.sin(t * np.pi * 4), 0)
            self.update_heatmap(self.current_state['security_check'][0], self.current_state['security_check'][1], 0.5 + 0.5 * np.sin(t * np.pi * 4), 'security_check')
            return {
                'positions': self.current_state.copy(),
                'colors': colors,
                'heatmap': self.heatmap_data.tolist(),
                'captured_states': self.captured_states
            }
        elif progress < 0.8:  # Phase 4: Measurement
            t = (progress - 0.6) * 5
            colors = self.colors.copy()
            colors['source'] = (0, 0, 1 - t)
            self.update_heatmap(self.current_state['source'][0], self.current_state['source'][1], 1-t, 'source')
            return {
                'positions': self.current_state.copy(),
                'colors': colors,
                'heatmap': self.heatmap_data.tolist(),
                'captured_states': self.captured_states
            }
        else:  # Phase 5: Teleportation completion
            t = (progress - 0.8) * 5
            colors = self.colors.copy()
            colors['target'] = (0, 1, t)  # Fade to bright green
            self.update_heatmap(self.current_state['target'][0], self.current_state['target'][1], t, 'target')
            return {
                'positions': self.current_state.copy(),
                'colors': colors,
                'heatmap': self.heatmap_data.tolist(),
                'captured_states': self.captured_states
            }

    def _interpolate(self, start: Tuple[float, float], end: Tuple[float, float], t: float) -> Tuple[float, float]:
        """Interpolate between two points"""
        return (
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t
        )

    def get_visualization_data(self) -> List[Dict]:
        """Generate visualization data for the complete teleportation process"""
        if self._frames is None:
            try:
                frames = []
                for i in range(100):
                    progress = i / 99.0
                    frames.append(self.get_state_snapshot(progress))
                self._frames = frames
                logger.info(f"Generated {len(frames)} visualization frames")
            except Exception as e:
                logger.error(f"Error generating visualization frames: {str(e)}")
                self._frames = []
        return self._frames

    def store_heatmap_data(self, teleportation_id: int):
        """Store current heat map data in the database"""
        try:
            # Store both heatmap grid data and captured states
            for state in self.captured_states:
                heatmap_point = HeatMapData(
                    teleportation_id=teleportation_id,
                    x_coord=state['x'],
                    y_coord=state['y'],
                    intensity=state['intensity'],
                    state_type=state['state_type']
                )
                db.session.add(heatmap_point)

            db.session.commit()
            logger.info(f"Stored heat map data for teleportation {teleportation_id}")
        except Exception as e:
            logger.error(f"Error storing heat map data: {str(e)}")
            db.session.rollback()


def get_visualization_frames():
    """Get visualization frames for the teleportation process"""
    try:
        visualizer = QuantumVisualizer()
        return visualizer.get_visualization_data()
    except Exception as e:
        logger.error(f"Error in get_visualization_frames: {str(e)}")
        return []

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    frames = get_visualization_frames()
    print(f"Generated {len(frames)} visualization frames")