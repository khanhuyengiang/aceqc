import numpy as np
from qutip import Qobj, basis, fidelity, average_gate_fidelity

# Ideal state |+><+|
plus_state = (basis(2,0) + basis(2,1)).unit()  # Creates the |+> state
minus_state = (basis(2,0) - basis(2,1)).unit()  # Creates the |-> state

rho_ideal = plus_state @ plus_state.dag()  # Ideal density matrix for the |+> state

# Ideal channel: S gate
S_gate = Qobj([[1,0],[0,1j]])  # S gate in matrix form

# Ideal measurement: X basis
plus = (basis(2,0) + basis(2,1)).unit()  # |+> state (basis for X measurement)
minus = (basis(2,0) - basis(2,1)).unit()  # |-> state (basis for X measurement)
M_ideal = plus * plus.dag()  # Measurement operator for the X basis

# Function to apply a measurement and return the result based on probability
def apply_measurement(rho, M):
    prob = (M * rho * M.dag()).tr()  # Probability of outcome from the measurement
    return 1 if prob > 0.5 else -1  # Return 1 or -1 depending on the measurement outcome

# Function to apply a measurement and return the raw probability of outcome
def apply_measurement_raw_prob(rho, M):
    prob = (M * rho * M.dag()).tr()  # Raw probability of the outcome
    return prob

# Function to apply a sequence of unitary gates to a density matrix
def apply_sequence(rho, sequence):
    """Apply a sequence of unitary gates to rho."""
    rho_out = rho
    for g in sequence:
        rho_out = g * rho_out * g.dag()  # Apply each gate in the sequence
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

# # Function to compute the average fidelity of the system and operations
# def average_fidelity(rho, gate, M):
#     avg_fid = (fidelity(rho, rho_ideal) + average_gate_fidelity(gate, S_gate) + fidelity(M, M_ideal)) / 3
#     if avg_fid < 0 or avg_fid > 1:
#         raise ValueError(f"Invalid average fidelity: {avg_fid}. Fidelity values must be between 0 and 1. ")
#     return round(avg_fid, 6)  # Average fidelity rounded to 6 decimal places

def average_fidelity(rho, gate, M):
    # Compute the individual fidelities for rho, gate, and M
    rho_fid = fidelity(rho, rho_ideal)
    gate_fid = average_gate_fidelity(gate, S_gate)
    M_fid = fidelity(M, M_ideal)

    # Check if any fidelity is out of the [0, 1] range
    if rho_fid < 0 or rho_fid > 1:
        print(rho)
        raise ValueError(f"Invalid fidelity for rho: {rho_fid}. Fidelity values must be between 0 and 1. "
                         "Check the rho fidelity calculation and the input state.")
    
    if gate_fid < 0 or gate_fid > 1:
        print(gate)
        raise ValueError(f"Invalid fidelity for gate: {gate_fid}. Fidelity values must be between 0 and 1. "
                         "Check the gate fidelity calculation and the input gate.")

    if M_fid < 0 or M_fid > 1:
        print(M)
        raise ValueError(f"Invalid fidelity for M: {M_fid}. Fidelity values must be between 0 and 1. "
                         "Check the M fidelity calculation and the input matrix M.")
    
    # Compute average fidelity
    avg_fid = (rho_fid + gate_fid + M_fid) / 3
    
    # Check if average fidelity is out of bounds
    if avg_fid < 0 or avg_fid > 1:
        raise ValueError(f"Invalid average fidelity: {avg_fid}. Average fidelity should be between 0 and 1. "
                         "This suggests there might be an error in the fidelity calculations.")

    return round(avg_fid, 6)  # Average fidelity rounded to 6 decimal places