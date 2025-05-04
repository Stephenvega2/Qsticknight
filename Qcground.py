import numpy as np
from scipy.integrate import odeint
from qiskit import Aer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.opflow import Z, I, PauliSumOp
from qiskit.utils import QuantumInstance
import matplotlib.pyplot as plt

# --- Step 1: Classical Grounding Simulation ---
def simulate_circuit_with_grounding(t, surge_voltage=10000):
    """
    Simulate control circuit handling a surge to get final voltage.
    t: time points (seconds)
    surge_voltage: Voltage of a surge
    Returns: Voltage over time
    """
    R = 50  # Resistance (ohms)
    C = 1e-6  # Capacitance (farads)
    R_ground = 10  # Grounding resistance (ohms)

    def circuit_dynamics(V, t):
        return -(V / (R * C)) - (V / R_ground)

    V0 = surge_voltage
    V = odeint(circuit_dynamics, V0, t)
    return V.flatten()

# --- Step 2: Qiskit Quantum Circuit (VQE) ---
def run_quantum_computation(final_voltage):
    """
    Run a VQE quantum circuit influenced by the grounding result (final voltage).
    final_voltage: Voltage from grounding simulation (used as a parameter)
    Returns: Ground state energy and optimal parameters
    """
    # Define a 2-qubit Hamiltonian (simple Ising model)
    hamiltonian = PauliSumOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)])

    # Set up the quantum simulator
    backend = Aer.get_backend("statevector_simulator")
    quantum_instance = QuantumInstance(backend)

    # Define the ansatz (quantum circuit)
    ansatz = TwoLocal(num_qubits=2, rotation_blocks=["ry"], entanglement_blocks="cz", reps=2)

    # Use final voltage to influence initial parameters (e.g., scale initial angles)
    initial_point = np.array([final_voltage * 0.1] * ansatz.num_parameters)  # Scale voltage to angles

    # Set up the optimizer
    optimizer = SPSA(maxiter=100)

    # Initialize VQE
    vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance, initial_point=initial_point)

    # Run VQE
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
    return result.eigenvalue.real, result.optimal_parameters

# --- Step 3: Classical Analysis of Quantum Results ---
def analyze_results(final_voltage, quantum_energy, optimal_params):
    """
    Analyze quantum results in the classical system.
    final_voltage: From grounding simulation
    quantum_energy: From VQE
    optimal_params: From VQE
    Returns: Dictionary with results
    """
    return {
        "final_voltage": final_voltage,
        "quantum_energy": quantum_energy,
        "optimal_parameters": optimal_params,
        "status": "Success" if abs(quantum_energy + 1.414) < 0.1 else "Check convergence"  # Rough check for expected energy
    }

# --- Run the Hybrid Workflow ---
t = np.linspace(0, 0.001, 1000)  # Time (seconds)

# Step 1: Run grounding simulation
surge_voltage = 10000  # 10,000V surge
voltages = simulate_circuit_with_grounding(t, surge_voltage)
final_voltage = voltages[-1]  # Final voltage after surge

# Step 2: Send to Qiskit for quantum computation
quantum_energy, optimal_params = run_quantum_computation(final_voltage)

# Step 3: Return to classical system for analysis
results = analyze_results(final_voltage, quantum_energy, optimal_params)

# --- Plot and Display Results ---
# Plot grounding simulation
plt.figure(figsize=(10, 6))
plt.plot(t * 1000, voltages, label="Circuit Voltage")
plt.title("Classical Grounding: Surge Dissipation (10,000V)")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.grid()
plt.legend()
plt.show()

# Print results
print("Hybrid Workflow Results:")
print(f"Final Voltage (Classical Grounding): {results['final_voltage']:.2f} V")
print(f"Quantum Ground State Energy (Qiskit VQE): {results['quantum_energy']:.3f}")
print(f"Optimal Quantum Parameters: {results['optimal_parameters']}")
print(f"Status: {results['status']}")

# Optional: Save results for further classical use
with open("hybrid_results.txt", "w") as f:
    f.write(str(results))
