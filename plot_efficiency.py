import time

import numpy as np
import matplotlib.pyplot as plt

from iteration import inverse_power_iteration, qr_iteration, lanczos_iteration

if __name__ == '__main__':
    # load hamiltonian
    hamiltonian = np.load('hamiltonian_size_80.npy')

    # similarity transformation
    np.random.seed(42)
    n = hamiltonian.shape[0]
    H = np.random.randn(n, n)
    Q, R = np.linalg.qr(H)
    modified_hamiltonian = Q @ hamiltonian @ Q.T

    # plot
    plt.figure(figsize=(6, 6), dpi=300)
    plt.title("Computation time")

    # comparison
    iteration_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # inverse power iteration - minimum eigenvalue
    inverse_power_iteration_time_list = list()
    for iteration in iteration_list:
        inverse_start_time = time.time()
        minimum_eigen_values_inv_power_iter = inverse_power_iteration(
            objective_matrix=modified_hamiltonian, num_iteration=iteration
        )
        inverse_end_time = time.time()
        computation_time = inverse_end_time - inverse_start_time
        inverse_power_iteration_time_list.append(computation_time)

    # plot
    plt.figure(figsize=(6, 6), dpi=300)
    plt.title("Computation time with matrix size 80")
    plt.plot(iteration_list, inverse_power_iteration_time_list, 'r', label='Inverse Power Iteration')

    # lanczos iteration
    lanczos_time_list = list()
    color = ['b']
    polynomial_degree_list = [29]
    for polynomial_degree, c in zip(polynomial_degree_list, color):
        lanczos_time_list = list()
        for iteration in iteration_list:
            lanczos_start_time = time.time()
            lanczos_T = lanczos_iteration(objective_matrix=modified_hamiltonian, polynomial_degree=polynomial_degree)
            lanczos_middle_time = time.time()
            lanczos_result = inverse_power_iteration(objective_matrix=lanczos_T, num_iteration=iteration)
            lanczos_end_time = time.time()
            lanczos_time_list.append(lanczos_end_time - lanczos_start_time)
        plt.plot(iteration_list, lanczos_time_list, c,
                 label='Lanczos iteration (dimension %d)' % (polynomial_degree + 1))

    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Time (unit:s)')
    plt.legend()
    plt.savefig('time.png')

    exit()

    # QR iteration -> get eigenvalue & eigenvector pairs
    eigen_value_matrix = qr_iteration(objective_matrix=modified_hamiltonian, num_iteration=20)
    # print(eigen_value_matrix.diagonal())



