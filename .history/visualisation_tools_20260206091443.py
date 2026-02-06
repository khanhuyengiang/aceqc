from qutip import Bloch

# Function to visualize random unitaries on the Bloch sphere
def visualize_unitaries_on_bloch(unitaries, plus_state, num_points=100):
    """
    Visualize random unitary operations applied to the |+> state on the Bloch sphere.

    Args:
    - unitaries: List of unitary operators to apply to the state.
    - plus_state: The initial state (should be a Qobj representing a qubit state).
    - num_points: Number of points (default is 100).
    """
    # Create a Bloch object for visualization
    bloch = Bloch()

    # Apply random unitaries to the initial state and store results
    for unitary in unitaries:
        # Apply the unitary transformation
        transformed_state = unitary * plus_state
        
        # Add transformed state to Bloch object
        bloch.add_states(transformed_state)

    # Show the Bloch sphere with the transformed states
    bloch.show()


# Function to visualize density matrices on the Bloch sphere
def visualize_dm_on_bloch(density_matrices):
    """
    Visualize density matrices on the Bloch sphere.

    Args:
    - density_matrices: List of density matrices (Qobj) to visualize.
    """
    # Create a Bloch object for visualization
    bloch = Bloch()

    # Add density matrices to Bloch object
    for rho in density_matrices:
        bloch.add_states(rho)

    # Show the Bloch sphere with the density matrices
    bloch.show()
