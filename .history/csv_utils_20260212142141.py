import csv
import os
import datetime
import itertools
from qsq_protocol import experiment_prob_failing,average_fidelity, rho_ideal,S_gate
from model_generator import generate_random_unitaries, generate_perturbed_rho, generate_perturbed_unitary, generate_uniform_rho
from qutip import Qobj, rand_dm, rand_ket, rand_unitary, ket2dm
def write_to_csv(csv_file, header, row):
    """
    Write a row of data to a CSV file. If the file doesn't exist, create it and write the header first.
    If the file exists, append the row to the existing file.

    Args:
        csv_file (str): The path to the CSV file.
        header (list): The header row to write if the file doesn't exist.
        row (list): The row of data to append or write to the CSV file.
    """
    # Check if the file exists
    if not os.path.isfile(csv_file):
        # If the file does not exist, create it and write the header first
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write the header
            writer.writerow(row)     # Write the first row of data
    else:
        # If the file exists, append the row
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)  # Append the row of data

def process_model_data(rho, gate, M, model_id):
    """
    Process the model data (rho, gate, M) and return a formatted row with calculated values.

    Args:
        rho (Qobj): The density matrix.
        gate (Qobj): The gate operator.
        M (Qobj): The measurement operator.
        model_id (str): The model ID.

    Returns:
        list: A row of data containing model_id, model data, P_vector, and model average fidelity.
    """
    # Normalize rho and M (ensure they're properly scaled)
    rho = rho / rho.tr()
    M = M / M.tr()

    # Convert the Qobjs to lists
    model = [rho.full().tolist(), gate.full().tolist(), M.full().tolist()]

    # Calculate P_vector and model average fidelity
    P_vector = experiment_prob_failing(rho, gate, M)
    model_average_fidelity = average_fidelity(rho, gate, M)

    # Create the row to return
    row = [model_id, model, P_vector, model_average_fidelity]

    return row


def generate_model_data(csv_file, method, N, **kwargs):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define the header for the CSV file
    header = ['model_id', 'model', 'P_vector', 'model_average_fidelity', 'model_average_fidelity_gauge']

    if method == "random":
        for i in range(N):
            model_id = timestamp + '_' + method + '_' + str(i)
            rho, M, gate = rand_dm(2), ket2dm(rand_ket(2)), rand_unitary(2)
            write_to_csv(csv_file,header,process_model_data(rho, gate, M, model_id))
    elif method == "perturbed":
        N = round(N**(1/3))
        perturbed_dm = generate_perturbed_rho(rho_ideal,N)
        perturbed_U = generate_perturbed_unitary(S_gate,N,2)
        perturbed_data = [(dm1, unitary, dm2) for dm1, unitary, dm2 in itertools.product(perturbed_dm, perturbed_U, perturbed_dm)]
        for i in range(len(perturbed_data)):
            model_id = timestamp + '_' + method + '_' + str(i)
            rho = perturbed_data[i][0]
            gate = perturbed_data[i][1]
            M = perturbed_data[i][2]
            write_to_csv(csv_file,header,process_model_data(rho, gate, M, model_id))
    elif method == 'uniform':
        N = round(N**(1/5))
        uniform_dm = generate_uniform_rho(N, N) 
        uniform_U = generate_random_unitaries(N)
        uniform_data = [(dm1, unitary, dm2) for dm1, unitary, dm2 in itertools.product(uniform_dm, uniform_U, uniform_dm)]
        for i in range(len(uniform_data)):
            model_id = timestamp + '_' + method + '_' + str(i)
            rho = uniform_data[i][0]
            gate = uniform_data[i][1]
            M = uniform_data[i][2]
            write_to_csv(csv_file,header,process_model_data(rho, gate, M, model_id))

def load_model_data(csv_file, load_gauge_fidelity=False):
    import csv
    from qutip import Qobj

    # Initialize lists
    model_ids = []
    models = []
    P_vectors = []
    model_average_fidelities = []
    model_average_fidelity_gauges = [] if load_gauge_fidelity else None

    # Open CSV
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)

        if load_gauge_fidelity:
            # Include gauge fidelities
            for row in reader:
                model_ids.append(row['model_id'])
                models.append([Qobj(m) for m in eval(row['model'])])
                P_vectors.append(eval(row['P_vector']))
                model_average_fidelities.append(float(row['model_average_fidelity']))
                model_average_fidelity_gauges.append(float(row['model_average_fidelity_gauges']))
        else:
            # Skip gauge fidelities
            for row in reader:
                model_ids.append(row['model_id'])
                models.append([Qobj(m) for m in eval(row['model'])])
                P_vectors.append(eval(row['P_vector']))
                model_average_fidelities.append(float(row['model_average_fidelity']))

    # Construct result dictionary
    result = {
        "model_ids": model_ids,
        "models": models,
        "P_vectors": P_vectors,
        "model_average_fidelities": model_average_fidelities,
    }

    if load_gauge_fidelity:
        result["model_average_fidelity_gauges"] = model_average_fidelity_gauges

    return result
