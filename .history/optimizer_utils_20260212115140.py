import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from qutip import Qobj
from qsq_protocol import average_fidelity
# Assuming you have compute_U, average_fidelity, decompose_U, plus_state, and minus_state defined
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
# Define the objective function to minimize

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
    # cos_theta2 = np.cos(theta2)
    theta1 = np.angle(U[0, 0])  # The phase angle of U[0, 0] gives θ1
    theta3 = np.angle(U[0, 1])  # The phase angle of U[0, 1] gives θ3
    
    return theta1, theta2, theta3
# Define the objective function to minimize
def objective_function(params, rho_exp, gate_exp, M_exp):
    # Unpack the parameters (theta1, theta2, theta3)
    theta1, theta2, theta3 = params
    
    # Compute the unitary operator U based on theta1, theta2, theta3
    U = compute_U(theta1, theta2, theta3)
    
    # Compute the objective function (average fidelity)
    return 1 - average_fidelity(U @ rho_exp @ U.dag(), U @ gate_exp @ U.dag(), U @ M_exp @ U.dag())

# Define optimization function
def optimize_method(method, objective_function, initial_guess, rho_exp, gate_exp, M_exp, bounds=[(-np.pi, np.pi), (0, np.pi/2), (-np.pi, np.pi)]):
    if method == 'BFGS' or method == 'COBYLA':
        result = minimize(
            objective_function, 
            initial_guess, 
            bounds=bounds,  
            args=(rho_exp, gate_exp, M_exp),  
            method=method
        )
    elif method == 'differential_evolution':
        result = differential_evolution(
            objective_function, 
            bounds=bounds,
            args=(rho_exp, gate_exp, M_exp),
            maxiter=2000,
            popsize=30
        )
    elif method == 'basinhopping':
        result = basinhopping(
            objective_function, 
            initial_guess,  
            minimizer_kwargs={"method": "COBYLA", "args": (rho_exp, gate_exp, M_exp)},  
        )
    else:
        raise ValueError("Unknown optimization method.")
    return result
