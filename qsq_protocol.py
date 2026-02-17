import numpy as np
from qutip import Qobj, basis, fidelity, average_gate_fidelity, dnorm 

plus_state = (basis(2,0) + basis(2,1)).unit()  # Creates the |+> state
minus_state = (basis(2,0) - basis(2,1)).unit()  # Creates the |-> state

# Ideal state |+><+|
rho_ideal = plus_state @ plus_state.dag()

# Ideal channel: S gate
S_gate = Qobj([[1,0],[0,1j]])  # S gate in matrix form

# Ideal measurement: X basis
M_ideal = plus_state @ plus_state.dag()  # Measurement operator for the X basis

# Function to apply a measurement and return the probability of outcome
def apply_measurement_raw_prob(rho, M):
    return (M @ rho @ M.dag()).tr()

# Function to apply a sequence of unitary gates to a density matrix
def apply_sequence(rho, sequence):
    """Apply a sequence of unitary gates to rho."""
    rho_out = rho
    for g in sequence:
        rho_out = g @ rho_out @ g.dag()  # Apply each gate in the sequence
    return rho_out

# Function to run the full experiment and return the raw probabilities of measurement outcomes
def experiment_raw_prob(rho, gate, M):
    sequences = [
        [],                        # ε (no gates applied)
        [gate, gate],              # SS (two S gates applied)
        [gate, gate, gate, gate]   # SSSS (four S gates applied)
    ]

    results = []
    for sequence in sequences:
        rho_out = apply_sequence(rho, sequence)  # Apply the gate sequence
        result = apply_measurement_raw_prob(rho_out, M)  # Measure outcome probability
        results.append(result)

    return results

# Function to calculate the "failing probability" (difference from ideal results)
def experiment_prob_failing(rho, gate, M):
    results = experiment_raw_prob(rho, gate, M)  # Get the raw probabilities
    ideal_results = [1, 0, 1]  # Ideal outcomes (for SS and SSSS sequences)
    prob_failing = [round(abs(ideal_results[i] - results[i]), 6) for i in range(len(results))]  # Difference from ideal
    return prob_failing

def operator_norm(A,B):
    C = (A - B).full()
    eigvals = np.linalg.eigvals(C.conj().T @ C)
    return (np.sqrt(np.max(eigvals))).real
  
# Function to compute the average fidelity of the system and operations
def average_fidelity(rho, gate, M):
    # Compute the fidelities
    fid_rho = fidelity(rho, rho_ideal)
    fid_gate = average_gate_fidelity(gate, S_gate)
    fid_M = 1 - operator_norm(M,M_ideal)

    # Check the validity of each fidelity value
    if not (0 <= round(fid_rho, 6) <= 1):
        raise ValueError(f"Invalid fidelity for rho: {fid_rho}. Fidelity values must be between 0 and 1.")
    
    if not (0 <= round(fid_gate, 6) <= 1):
        raise ValueError(f"Invalid fidelity for gate: {fid_gate}. Average gate fidelity values must be between 0 and 1.")
    
    if not (0 <= round(fid_M, 6) <= 1):
        raise ValueError(f"Invalid operator distance for M: {fid_M}. Operator distance values must be between 0 and 1.")
    
    # Calculate the average fidelity
    avg_fid = (fid_rho + fid_gate + fid_M) / 3
    
    if not (0 <= round(avg_fid, 6) <= 1):
        raise ValueError(f"Invalid average fidelity: {avg_fid}. Average fidelity must be between 0 and 1.")
    
    return avg_fid


