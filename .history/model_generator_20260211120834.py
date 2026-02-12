import numpy as np
import qutip as qt
from qutip import Qobj, rand_herm, rand_unitary, qeye, sigmax, sigmay, sigmaz

# Function to generate perturbed density matrices
def generate_perturbed_rho(rho_ideal,N):
    """
    Generates perturbed density matrices by applying random Hermitian perturbations.
    
    Args:
    - rho_ideal: The ideal initial density matrix.
    - perturbation_strengths: List of strengths for the perturbations.
    
    Returns:
    - List of perturbed density matrices.
    """
    perturbed_rhos = []  # List to store perturbed density matrices
    for i in range(N):
        # Step 1: Generate a Hermitian perturbation matrix
        perturbation = rand_herm(rho_ideal.shape[0])
        
        # Step 2: Apply the perturbation to the original density matrix
        perturbed_rho = rho_ideal + perturbation
        
        # Step 3: Ensure Hermiticity (average with its conjugate transpose)
        perturbed_rho = (perturbed_rho + perturbed_rho.dag()) / 2
        
        # Step 4: Ensure the density matrix is positive semi-definite
        eigenvalues = perturbed_rho.eigenenergies()
        if any(eigenvalue < 0 for eigenvalue in eigenvalues):
            # Add a small positive shift if there are negative eigenvalues
            perturbed_rho = perturbed_rho + abs(min(eigenvalues)) * qeye(rho_ideal.shape[0])
        
        # Step 5: Normalize to trace 1
        perturbed_rho = perturbed_rho / perturbed_rho.tr()

        # Step 6: Append the perturbed density matrix to the list
        perturbed_rhos.append(perturbed_rho)
    
    return perturbed_rhos


# Function to generate perturbed unitary operators
def generate_perturbed_unitary(U_ideal, num_samples, noise_range=0.1):
    """
    Generates perturbed unitary operators by adding random noise.
    
    Args:
    - U_ideal: The ideal initial unitary operator (Qobj).
    - num_samples: The number of perturbed unitary matrices to generate.
    - noise_range: The range of random noise to add to the unitary matrix.
    
    Returns:
    - List of perturbed unitary operators.
    """
    perturbed_unitaries = []
    
    for _ in range(num_samples):
        # Add random noise to the unitary matrix
        noise = noise_range * (np.random.rand(*U_ideal.shape) - 0.5)
        U_perturbed = U_ideal.full() + noise  # Convert to NumPy array and add noise
        
        # Perform SVD
        U, s, Vh = np.linalg.svd(U_perturbed)
        
        # Reconstruct the unitary matrix by setting singular values to 1
        U_unitary = np.dot(U, Vh)
        
        # Convert the resulting unitary matrix back to a Qobj
        U_unitary_qobj = Qobj(U_unitary)
        perturbed_unitaries.append(U_unitary_qobj)
    
    return perturbed_unitaries


# Function to generate density matrix for a given Bloch vector (r_x, r_y, r_z)
def bloch_vector_to_density_matrix(r):
    """
    Generate a density matrix corresponding to a given Bloch vector (r_x, r_y, r_z).
    
    Args:
    - r: A tuple (r_x, r_y, r_z) representing the components of the Bloch vector.
    
    Returns:
    - A density matrix (Qobj) corresponding to the Bloch vector.
    """
    I = qeye(2)  # Identity matrix
    sig_x, sig_y, sig_z = sigmax(), sigmay(), sigmaz()
    return (I + r[0] * sig_x + r[1] * sig_y + r[2] * sig_z) / 2


# Function to generate density matrices for a grid of points on the Bloch sphere
def generate_density_matrices(theta_steps, phi_steps):
    """
    Generate density matrices corresponding to a grid of points on the Bloch sphere.
    
    Args:
    - theta_steps: Number of points along the polar angle theta (0 to pi).
    - phi_steps: Number of points along the azimuthal angle phi (0 to 2pi).
    
    Returns:
    - A list of density matrices (Qobjs) corresponding to points on the Bloch sphere.
    """
    density_matrices = []
    
    # Loop over the polar angle theta (from 0 to pi)
    for i in range(theta_steps):
        theta = np.pi * i / (theta_steps - 1)  # Evenly spaced values of theta (0 to pi)
        
        # Loop over the azimuthal angle phi (from 0 to 2pi)
        for j in range(phi_steps):
            phi = 2 * np.pi * j / phi_steps  # Evenly spaced values of phi (0 to 2pi)
            
            # Compute the corresponding Bloch vector components
            r_x = np.sin(theta) * np.cos(phi)
            r_y = np.sin(theta) * np.sin(phi)
            r_z = np.cos(theta)
            
            # Create and append the density matrix
            density_matrices.append(bloch_vector_to_density_matrix((r_x, r_y, r_z)))
    
    return density_matrices


# Function to generate a list of uniformly random unitary operators (using Haar measure)
def generate_random_unitaries(num_units):
    """
    Generates random unitary matrices using the Haar measure.
    
    Args:
    - num_units: Number of random unitary matrices to generate.
    
    Returns:
    - List of random unitary matrices (Qobjs).
    """
    unitaries = []
    for _ in range(num_units):
        unitary = rand_unitary(2)  # Random unitary matrix of size 2x2 (for qubit)
        unitaries.append(unitary)
    return unitaries
