import csv
import os
import datetime
import itertools
from qsq_protocol import experiment_prob_failing, average_fidelity, rho_ideal, S_gate
from model_generator import (
    generate_random_unitaries,
    generate_perturbed_rho,
    generate_perturbed_unitary,
    generate_uniform_rho,
)
from qutip import Qobj, rand_dm, rand_ket, rand_unitary, ket2dm


def write_to_csv(csv_file, header, row):
    """
    Write a single row to a CSV file.
    Creates the file and writes the header if it does not exist,
    otherwise appends the row.
    """
    if not os.path.isfile(csv_file):
        # Create file and write header + first row
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(row)
    else:
        # Append row to existing file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)


def process_model_data(rho, gate, M, model_id):
    """
    Compute model statistics and return a CSV-ready row.
    """
    # Ensure proper normalization
    rho = rho / rho.tr()
    M = M / M.tr()

    # Store model components as plain lists
    model = [rho.full().tolist(), gate.full().tolist(), M.full().tolist()]

    # Compute failure probabilities and average fidelity
    P_vector = experiment_prob_failing(rho, gate, M)
    model_average_fidelity = average_fidelity(rho, gate, M)

    return [model_id, model, P_vector, model_average_fidelity]


def generate_model_data(csv_file, method, N, **kwargs):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # CSV column names
    header = [
        'model_id',
        'model',
        'P_vector',
        'model_average_fidelity',
        'model_average_fidelity_gauge',
    ]

    if method == "random":
        for i in range(N):
            model_id = f"{timestamp}_{method}_{i}"
            rho, M, gate = rand_dm(2), ket2dm(rand_ket(2)), rand_unitary(2)
            write_to_csv(csv_file, header, process_model_data(rho, gate, M, model_id))

    elif method == "perturbed":
        N = round(N ** (1 / 3))
        perturbed_dm = generate_perturbed_rho(rho_ideal, N)
        perturbed_U = generate_perturbed_unitary(S_gate, N, 2)

        perturbed_data = itertools.product(perturbed_dm, perturbed_U, perturbed_dm)
        for i, (rho, gate, M) in enumerate(perturbed_data):
            model_id = f"{timestamp}_{method}_{i}"
            write_to_csv(csv_file, header, process_model_data(rho, gate, M, model_id))

    elif method == "uniform":
        N = round(N ** (1 / 5))
        uniform_dm = generate_uniform_rho(N, N)
        uniform_U = generate_random_unitaries(N)

        uniform_data = itertools.product(uniform_dm, uniform_U, uniform_dm)
        for i, (rho, gate, M) in enumerate(uniform_data):
            model_id = f"{timestamp}_{method}_{i}"
            write_to_csv(csv_file, header, process_model_data(rho, gate, M, model_id))


def load_model_data(csv_file):
    # Containers for each column
    model_ids = []
    models = []
    P_vectors = []
    model_average_fidelities = []
    model_average_fidelity_gauges = []

    # Read CSV contents
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            model_ids.append(row['model_id'])

            # Reconstruct Qobj model components
            model_data = eval(row['model'])
            qobj_model = [Qobj(m) for m in model_data]
            models.append(qobj_model)

            # Parse stored values
            P_vectors.append(eval(row['P_vector']))
            model_average_fidelities.append(float(row['model_average_fidelity']))

            # Gauge fidelity may not always be present
            if 'model_average_fidelity_gauge' in row and 'model_average_fidelity_gauge' is not None:
                model_average_fidelity_gauges.append(
                    float(row['model_average_fidelity_gauge'])
                )
