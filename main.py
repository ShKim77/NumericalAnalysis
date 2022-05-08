import numpy as np

from scipy.linalg import qr

from util import get_hamiltonian_ij
from iteration import power_iteration, inverse_power_iteration, inverse_shift_power_iteration
from iteration import orthogonal_iteration, qr_iteration, lanczos_iteration


if __name__ == '__main__':
    # get hermitian matrix
    # n_list = [4, 20, 40, 80, 160, 320, 640, 1280]
    # for n in n_list:
    #     hamiltonian = np.zeros(shape=(n, n))
    #     for row_idx in range(n):
    #         for col_idx in range(n):
    #             hamiltonian[row_idx, col_idx] = get_hamiltonian_ij(
    #                 func_i_degree=row_idx, func_j_degree=col_idx, h=1, m=1, angular_w=1
    #             )
    #     print(hamiltonian)
    #     exit()
    #     np.save('hamiltonian_size_%d' % n, hamiltonian)
    # exit()
    n = 4
    hamiltonian = np.zeros(shape=(n, n))
    for row_idx in range(n):
        for col_idx in range(n):
            hamiltonian[row_idx, col_idx] = get_hamiltonian_ij(
                func_i_degree=row_idx, func_j_degree=col_idx, h=1, m=1, angular_w=1
            )

    # similarity transformation
    np.random.seed(42)
    H = np.random.randn(n, n)

    # Basis transformation.
    # Let qsi -> eigenbasis (hermite function) and phi -> non eigenbasis
    # then phi = Q @ qsi
    Q, R = qr(H)

    # real answer (in terms of Hermite functions) -> Q.T @ (eigenvectors from modified hamiltonian)
    # modified hamiltonian is symmetric positive definite
    modified_hamiltonian = Q @ hamiltonian @ Q.T

    print(modified_hamiltonian)
    exit()
    # power iteration - maximum eigenvalue
    maximum_eigen_values_power_iter = power_iteration(objective_matrix=modified_hamiltonian, num_iteration=20)

    # inverse power iteration - minimum eigenvalue
    minimum_eigen_values_inv_power_iter = inverse_power_iteration(
        objective_matrix=modified_hamiltonian, num_iteration=20
    )

    # inverse iteration with shift - nearest eigenvalue to the shift
    nearest_eigen_value_to_the_target = inverse_shift_power_iteration(
        objective_matrix=modified_hamiltonian, shift=2.2, num_iteration=10
    )

    # orthogonal iteration -> prevent eigenvectors from becoming ill-conditioned -> get eigenvectors
    p = 4
    eigenvector_matrix = orthogonal_iteration(p=p, objective_matrix=modified_hamiltonian, num_iteration=100)

    # check eigenvalue
    hermite_eigen_vectors = Q.T @ eigenvector_matrix
    for col_idx in range(p):
        norm = np.linalg.norm(hermite_eigen_vectors[:, col_idx])
        hermite_eigen_vectors[:, col_idx] = hermite_eigen_vectors[:, col_idx] / norm
    print(hermite_eigen_vectors)  # all columns go to the dominant eigenvectors

    # QR iteration -> get eigenvalue & eigenvector pairs
    eigen_value_matrix = qr_iteration(objective_matrix=modified_hamiltonian, num_iteration=50)
    print(eigen_value_matrix.diagonal())

    # Lanczos iteration
    result = lanczos_iteration(objective_matrix=modified_hamiltonian, polynomial_degree=5, num_power_iteration=20,
                               value_type='maximum')

