import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from qutip import Qobj
from qsq_protocol import average_fidelity, plus_state, minus_state

def compute_U(theta1, theta2, theta3):
    # Build a single-qubit unitary from three angles
    U = np.array([
        [np.exp(1j * theta1) * np.cos(theta2), np.exp(1j * theta3) * np.sin(theta2)],
        [np.exp(-1j * theta3) * np.sin(theta2), np.exp(-1j * theta1) * np.cos(theta2)]
    ])
    return Qobj(U).unit()


def decompose_U(U):
    # Recover angles from a given unitary matrix
    theta2 = np.arcsin(np.abs(U[1, 0]))  # From off-diagonal magnitude
    theta1 = np.angle(U[0, 0])           # Phase of (0,0) element
    theta3 = np.angle(U[0, 1])           # Phase of (0,1) element
    return theta1, theta2, theta3


def objective_function(params, rho_exp, gate_exp, M_exp):
    # Parameters are (theta1, theta2, theta3)
    theta1, theta2, theta3 = params

    # Apply the unitary to the experimental objects
    U = compute_U(theta1, theta2, theta3)

    # Minimize one minus the average fidelity
    return 1 - average_fidelity(
        U @ rho_exp @ U.dag(),
        U @ gate_exp @ U.dag(),
        U @ M_exp @ U.dag()
    )


def optimize_method(
    method,
    objective_function,
    initial_guess,
    rho_exp,
    gate_exp,
    M_exp,
    bounds=[(-np.pi, np.pi), (0, np.pi / 2), (-np.pi, np.pi)]
):
    # Choose optimization backend
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
            minimizer_kwargs={
                "method": "COBYLA",
                "args": (rho_exp, gate_exp, M_exp)
            },
        )

    else:
        raise ValueError("Unknown optimization method.")

    return result

def average_fidelity_gauge(rho, gate, M, avg_fidelity=None):
    """
    Compute the gauge-corrected average fidelity.
    """
    if avg_fidelity is None:
        avg_fidelity = average_fidelity(rho,gate,M)
    
    eigenvalues, eigenstates = M.eigenstates()
    U_guess = eigenstates[0] @ plus_state.dag() - 1j * eigenstates[1] @ minus_state.dag()
    initial_guess = decompose_U(U_guess)
    bounds = [(-np.pi, np.pi), (0, np.pi/2), (-np.pi, np.pi)]
    result_C = optimize_method('basinhopping', objective_function, initial_guess, rho, gate, M, bounds)
    result_D = optimize_method('differential_evolution', objective_function, initial_guess, rho, gate, M, bounds)
    result = max(1-result_C.fun, 1-result_D.fun, avg_fidelity)
    
    return result