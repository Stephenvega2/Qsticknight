import logging
import unittest
from quantum_teleportation import QuantumTeleporter, run_demo
from datetime import datetime
import numpy as np
from app import app  # Import Flask app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestQuantumTeleportation(unittest.TestCase):
    def setUp(self):
        """Initialize teleporter for each test"""
        self.teleporter = QuantumTeleporter(num_shots=1000)
        self.app = app
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after each test"""
        self.app_context.pop()

    def test_basic_teleportation(self):
        """Test basic quantum teleportation functionality"""
        states = ['0', '1', '+', 'i']
        for state in states:
            experience = self.teleporter.perform_teleportation(state)
            self.assertGreater(experience['metrics']['success_probability'], 0.2,
                             f"Low success probability for state {state}")
            self.assertGreater(experience['metrics']['channel_security'], 0.15,
                             f"Low channel security for state {state}")

    def test_security_breach_detection(self):
        """Test security breach detection system"""
        experience = self.teleporter.perform_teleportation('+')
        self.assertIn('security_status', experience)
        self.assertIn('status', experience['security_status'])

        # Verify security metrics with adjusted thresholds
        self.assertGreater(experience['metrics']['security_level'], 0.3,
                          "Security level below threshold")
        self.assertGreater(experience['metrics']['entropy_stability'], 0.3,
                          "Low entropy stability")

    def test_temperature_management(self):
        """Test temperature control and monitoring"""
        experience = self.teleporter.perform_teleportation('0')
        self.assertLessEqual(experience['temperature'], 
                           self.teleporter.temperature_params['max_temp'],
                           "Temperature exceeds maximum limit")

        # Test temperature increase with operations
        initial_temp = self.teleporter.temperature_params['current_temp']
        self.teleporter.update_temperature(100)  # Simulate heavy load
        self.assertGreater(self.teleporter.temperature_params['current_temp'],
                          initial_temp,
                          "Temperature not increasing with operations")

    def test_noise_resistance(self):
        """Test noise resistance and near-zero state handling"""
        # Test near-zero states
        zero_state = self.teleporter.create_near_zero_state('standard')
        self.assertTrue(np.allclose(zero_state, np.array([0, 0, 0])),
                       "Invalid standard near-zero state")

        # Test noise resistance
        noisy_state = np.array([0.7, 0.3, 0.2, 0.1])
        corrected_state = self.teleporter.apply_noise_resistance(noisy_state)
        self.assertLess(np.linalg.norm(corrected_state - noisy_state), 0.5,
                       "Noise correction too aggressive")

    def test_reward_system(self):
        """Test user reward and achievement system"""
        experience = self.teleporter.perform_teleportation('+')
        self.assertIn('contribution_points', experience['metrics'])
        self.assertIn('trust_level', experience['metrics'])
        self.assertIn('trust_level_name', experience['metrics'])

        # Verify point calculation
        self.assertGreater(experience['metrics']['contribution_points'], 0,
                          "No contribution points awarded")

def run_audit():
    """Run comprehensive security and UI audit"""
    logger.info("Starting Quantum Teleportation Security and UI Audit")

    # Run all tests with app context
    with app.app_context():
        suite = unittest.TestLoader().loadTestsFromTestCase(TestQuantumTeleportation)
        result = unittest.TextTestRunner(verbosity=2).run(suite)

        # Generate audit summary
        audit_summary = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
        }

        logger.info("\nAudit Summary:")
        logger.info(f"Tests Run: {audit_summary['tests_run']}")
        logger.info(f"Success Rate: {audit_summary['success_rate']*100:.1f}%")
        logger.info(f"Failures: {audit_summary['failures']}")
        logger.info(f"Errors: {audit_summary['errors']}")

        return audit_summary

if __name__ == "__main__":
    print("Running Quantum Teleportation Security and UI Audit...")
    with app.app_context():
        audit_results = run_audit()
        print("\nRunning Full Demo...")
        run_demo()