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

# Function to compute the average fidelity of the system and operations
def average_fidelity(rho, gate, M):
    avg_fid = (fidelity(rho.unit(), rho_ideal) + average_gate_fidelity(gate.unit(), S_gate) + fidelity(M.unit(), M_ideal)) / 3
    avg_fid = round(avg_fid, 6) # Average fidelity rounded to 6 decimal places
    if avg_fid < 0 or avg_fid > 1:
        raise ValueError(f"Invalid average fidelity: {avg_fid}. Fidelity values must be between 0 and 1. ")
    return  avg_fid 

from scipy.optimize import differential_evolution
from scipy.optimize import minimize
# Define the objective function to minimize
def objective_function(params, rho_exp, gate_exp, M_exp):
    # Unpack the parameters (theta1, theta2, theta3)
    theta1, theta2, theta3 = params
    
    # Compute the unitary operator U based on theta1, theta2, theta3
    U = compute_U(theta1, theta2, theta3)
    
    # Compute the objective function (average fidelity)
    return 1 - average_fidelity(U @ rho_exp @ U.dag(), U @ gate_exp @  U.dag(), U @ M_exp @  U.dag())

def average_fidelity_gauge(rho,gate,M):
    # Initial guess for the parameters (theta1, theta2, theta3)
    eigenvalues, eigenstates = M.eigenstates()
    U_guess = eigenstates[0] @ plus_state.dag() - 1j * eigenstates[1] @ minus_state.dag()
    initial_guess = decompose_U(U_guess)

    # Perform the optimization
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(rho, gate, M),  # Pass the fixed matrices as extra arguments
        method='COBYLA'  # You can change this to another method if needed
    )

    initial_fid = average_fidelity(rho,gate,M)
    max_fid = 1-result.fun
    if max_fid < 1.1*initial_fid:
        result_global = differential_evolution(
            objective_function, 
            bounds=[(0, 2*np.pi), (0, 0.5*np.pi/2), (0, 2*np.pi)],  # adjust bounds based on the physical parameters
            args=(rho, gate, M),
            strategy='best1bin',  # You can tweak the strategy
            maxiter=1000,
            popsize=20  # Try different population sizes if needed
        )
        max_fid = max(1-result.fun, 1-result_global.fun,average_fidelity(rho,gate,M))
    return max_fid

def compute_U(theta1, theta2, theta3):
    # Constructing the unitary matrix U(θ1, θ2, θ3)
    U = np.array([
        [np.exp(1j * theta1) * np.cos(theta2), np.exp(1j * theta3) * np.sin(theta2)],
        [np.exp(-1j * theta3) * np.sin(theta2), np.exp(-1j * theta1) * np.cos(theta2)]
    ])
    return Qobj(U).unit()

# Function 2: Given a matrix U, compute theta1, theta2, theta3
def decompose_U(U):
    # Extracting theta2 from the sine elements (sin(θ2) term)
    theta2 = np.arcsin(np.abs(U[1, 0]))  # Using the off-diagonal elements
    
    # Using the real part of U[0, 0] and U[1, 1] to extract theta1 and theta3
    cos_theta2 = np.cos(theta2)
    theta1 = np.angle(U[0, 0])  # The phase angle of U[0, 0] gives θ1
    theta3 = np.angle(U[0, 1])  # The phase angle of U[0, 1] gives θ3
    
    return theta1, theta2, theta3