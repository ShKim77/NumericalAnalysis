import numpy as np
import matplotlib.pyplot as plt

from iteration import inverse_power_iteration, qr_iteration, lanczos_iteration

if __name__ == '__main__':
    # load hamiltonian
    hamiltonian_10 = np.load('hamiltonian_size_10.npy')
    hamiltonian_20 = np.load('hamiltonian_size_20.npy')
    hamiltonian_40 = np.load('hamiltonian_size_40.npy')
    hamiltonian_80 = np.load('hamiltonian_size_80.npy')

    hamiltonian_list = [hamiltonian_10, hamiltonian_20, hamiltonian_40, hamiltonian_80]
    num_iteration = 100
    matrix_size = [10, 20, 40, 80]
    error = list()
    for hamiltonian in hamiltonian_list:
        # similarity transformation
        np.random.seed(42)

        # for hamiltonian in hamiltonian_list:
        n = hamiltonian.shape[0]
        H = np.random.randn(n, n)
        Q, R = np.linalg.qr(H)
        modified_hamiltonian = Q @ hamiltonian @ Q.T

        # inverse transformation
        minimum_eigen_values_inv_power_iter = inverse_power_iteration(
            objective_matrix=modified_hamiltonian, num_iteration=num_iteration
        )
        inverse_power_iteration_error = np.abs(np.array(minimum_eigen_values_inv_power_iter) - 0.5)
        inverse_power_iteration_error /= 0.5
        error.append(inverse_power_iteration_error[-1])

    # plot
    plt.figure(figsize=(6, 6), dpi=300)
    plt.title("Ground state energy calculation (matrix size 20)")
    plt.plot(matrix_size, error, 'ro', label='Inverse Power Iteration')
    plt.plot(matrix_size, error, 'r')
    plt.hlines(y=1e-16, xmin=1, xmax=80, color='black', linestyles='--', label='Machine precision ($10^{-16}$)')
    plt.xticks(ticks=matrix_size)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Relative Error', fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig('error.png')
    exit()

    # comparison
    plt.figure(figsize=(6, 6), dpi=300)
    plt.title("Ground state energy calculation (matrix size 20)")
    num_iteration = 20
    polynomial_degree_list = [4, 9, 14, 19]
    color = ['r', 'g', 'b', 'm']
    for polynomial_degree, c in zip(polynomial_degree_list, color):
        lanczos_result = lanczos_iteration(objective_matrix=modified_hamiltonian, polynomial_degree=polynomial_degree)
        lanczos_result = inverse_power_iteration(objective_matrix=lanczos_result, num_iteration=num_iteration)
        lanczos_error = np.abs(np.array(lanczos_result) - 0.5) / 0.5
        plt.plot(range(1, num_iteration + 1), lanczos_error, c,
                 label='Krylov subspace dimension = %d' % (polynomial_degree + 1))

    plt.yscale('log')
    # plt.yticks(ticks=[1e-16])
    plt.xticks(ticks=[1, 5, 10, 15, 20])
    # plt.hlines(y=1e-16, xmin=1, xmax=80, color='black', linestyles='--', label='Machine precision ($10^{-16}$)')
    # plt.vlines()
    # plt.hlines(y=1e-16, xmin=1, xmax=80, color='black', linestyles='--', label='Machine precision ($10^{-16}$)')
    # plt.xlabel('log')
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Relative Error', fontsize=10)
    plt.legend(fontsize=10, loc='lower left')
    plt.savefig('error.png')
    exit()




    # inverse power iteration - minimum eigenvalue
    minimum_eigen_values_inv_power_iter = inverse_power_iteration(
        objective_matrix=modified_hamiltonian, num_iteration=num_iteration
    )
    inverse_power_iteration_error = np.abs(np.array(minimum_eigen_values_inv_power_iter) - 0.5)
    inverse_power_iteration_error /= 0.5

    # lanczos iteration
    polynomial_degree = 9
    lanczos_result = lanczos_iteration(objective_matrix=modified_hamiltonian, polynomial_degree=polynomial_degree)
    lanczos_result = inverse_power_iteration(objective_matrix=lanczos_result, num_iteration=num_iteration)
    lanczos_error = np.abs(np.array(lanczos_result) - 0.5) / 0.5

    plt.figure(figsize=(6, 6), dpi=300)
    plt.title("Ground state energy calculation (matrix size 20)")
    plt.plot(range(1, num_iteration + 1), lanczos_error, 'g',
             label='Krylov subspace dimension = %d' % (polynomial_degree + 1))
    plt.plot(range(1, num_iteration + 1), inverse_power_iteration_error, 'r', label='Inverse power iteration')

    polynomial_degree = 14
    lanczos_result = lanczos_iteration(objective_matrix=modified_hamiltonian, polynomial_degree=polynomial_degree)
    lanczos_result = inverse_power_iteration(objective_matrix=lanczos_result, num_iteration=num_iteration)
    lanczos_error = np.abs(np.array(lanczos_result) - 0.5) / 0.5
    plt.plot(range(1, num_iteration + 1), lanczos_error, 'b',
             label='Krylov subspace dimension = %d' % (polynomial_degree + 1))
    # plt.plot(range(1, num_iteration + 1), lanczos_error, 'r', label='Lanczos iteration (Krylov subspace dimension = %d)' % (polynomial_degree + 1))
    # plt.plot(size, error_list, 'ro', label='Inverse Power Iteration')

    # inverse power iteration - minimum eigenvalue
    # minimum_eigen_values_inv_power_iter = inverse_power_iteration(
    #     objective_matrix=modified_hamiltonian_list[1], num_iteration=num_iteration
    # )
    # inverse_power_iteration_error = np.abs(np.array(minimum_eigen_values_inv_power_iter) - 0.5)
    # inverse_power_iteration_error /= 0.5
    # lanczos iteration
    # polynomial_degree = 39
    # lanczos_result = lanczos_iteration(objective_matrix=modified_hamiltonian, polynomial_degree=polynomial_degree)
    # lanczos_result = inverse_power_iteration(objective_matrix=lanczos_result, num_iteration=num_iteration)
    # lanczos_error = np.abs(np.array(lanczos_result) - 0.5)

    # plt.plot(range(1, num_iteration + 1), inverse_power_iteration_error, 'r', label='Inverse Power Iteration (matrix size = 20)')
    # plt.plot(range(1, num_iteration + 1), lanczos_error, 'b',
    #          label='Lanczos iteration (dimension %d)' % (polynomial_degree + 1))
    plt.yscale('log')
    # plt.yticks(ticks=[1e-16])
    plt.xticks(ticks=[1, 5, 10, 15, 20])
    # plt.vlines()
    # plt.hlines(y=1e-16, xmin=1, xmax=80, color='black', linestyles='--', label='Machine precision ($10^{-16}$)')
    # plt.xlabel('log')
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Relative Error', fontsize=10)
    plt.legend(fontsize=10, loc='lower left')
    plt.savefig('error.png')

    exit()

    # QR iteration -> get eigenvalue & eigenvector pairs
    eigen_value_matrix = qr_iteration(objective_matrix=modified_hamiltonian, num_iteration=20)
    # print(eigen_value_matrix.diagonal())

    # plot
    plt.figure(figsize=(6, 6), dpi=300)
    plt.title("Ground state energy calculation - Lanczos iteration")

