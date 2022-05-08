import numpy as np
np.random.seed(42)


def power_iteration(objective_matrix, num_iteration):
    """
    Get maximum eigenvalue
    """
    # get dimension
    if objective_matrix.shape[0] != objective_matrix.shape[1]:
        raise ValueError("Insert square hermitian matrix")
    matrix_dimension = objective_matrix.shape[0]

    # power iteration
    initial_vector = np.random.randn(matrix_dimension)
    before_x = initial_vector / np.linalg.norm(initial_vector)
    ratio_list = list()
    for num in range(num_iteration):
        # multiplication
        next_x = objective_matrix @ before_x

        # get ratio
        maximum_eigenvalue = np.linalg.norm(next_x) / np.linalg.norm(before_x)
        ratio_list.append(maximum_eigenvalue)

        # normalized
        next_x /= np.linalg.norm(next_x)

        # store
        before_x = next_x

    return ratio_list


def inverse_power_iteration(objective_matrix, num_iteration):
    """
    Get mimimum eigenvalue
    """
    # get dimension
    if objective_matrix.shape[0] != objective_matrix.shape[1]:
        raise ValueError("Insert square hermitian matrix")
    matrix_dimension = objective_matrix.shape[0]

    # inverse power iteration
    initial_vector = np.random.randn(matrix_dimension)
    before_x = initial_vector / np.linalg.norm(initial_vector)
    ratio_list = list()
    for num in range(num_iteration):
        next_x = np.linalg.solve(objective_matrix, before_x)
        minimum_eigenvalue = 1 / np.linalg.norm(next_x)
        ratio_list.append(minimum_eigenvalue)
        next_x /= np.linalg.norm(next_x)
        before_x = next_x

    return ratio_list


def inverse_shift_power_iteration(objective_matrix, shift, num_iteration):
    """
    Get nearest eigenvalue to the target
    """
    # get dimension
    if objective_matrix.shape[0] != objective_matrix.shape[1]:
        raise ValueError("Insert Square Hermitian Matrix")
    matrix_dimension = objective_matrix.shape[0]

    # inverse shift power iteration
    initial_vector = np.random.randn(matrix_dimension)
    before_x = initial_vector / np.linalg.norm(initial_vector)
    shifted_objective_matrix = objective_matrix - shift * np.eye(matrix_dimension)

    ratio_list = list()
    for num in range(num_iteration):
        # print(np.linalg.norm(before_x))
        next_x = np.linalg.solve(shifted_objective_matrix, before_x)
        # print(np.linalg.norm(next_x))
        sign = np.sign(next_x @ before_x)
        minimum_eigenvalue = sign * 1 / np.linalg.norm(next_x) + shift
        ratio_list.append(minimum_eigenvalue)
        next_x /= np.linalg.norm(next_x)
        before_x = next_x

    return ratio_list


def simultaneous_iteration(p, objective_matrix, num_iteration):
    """
    eigenvectors became ill-conditioned (parallel)
    """
    # get dimension
    if objective_matrix.shape[0] != objective_matrix.shape[1]:
        raise ValueError("Insert Square Hermitian Matrix")
    matrix_dimension = objective_matrix.shape[0]

    X_initial = np.random.randn(matrix_dimension, p)
    # normalize
    for col_idx in range(p):
        norm = np.linalg.norm(X_initial[:, col_idx])
        X_initial[:, col_idx] = X_initial[:, col_idx] / norm

    X_before = X_initial.copy()
    for num in range(num_iteration):
        X_next = objective_matrix @ X_before
        # normalize
        for col_idx in range(p):
            norm = np.linalg.norm(X_next[:, col_idx])
            X_next[:, col_idx] = X_next[:, col_idx] / norm
        # store
        X_before = X_next

    return X_before


def orthogonal_iteration(p, objective_matrix, num_iteration):
    """
    Prevent eigenvectors from becoming ill-conditioned
    """
    # get dimension
    if objective_matrix.shape[0] != objective_matrix.shape[1]:
        raise ValueError("Insert Square Hermitian Matrix")
    matrix_dimension = objective_matrix.shape[0]

    X_before = np.random.randn(matrix_dimension, p)
    for num in range(num_iteration):
        Q, R = np.linalg.qr(X_before)
        X_next = objective_matrix @ Q
        X_before = X_next

    return X_before


def qr_iteration(objective_matrix, num_iteration):
    """
    QR iteration -> get eigenvalue & eigenvector pairs
    return: eigen value matrix
    """
    # get dimension
    if objective_matrix.shape[0] != objective_matrix.shape[1]:
        raise ValueError("Insert Square Hermitian Matrix")
    matrix_dimension = objective_matrix.shape[0]

    X_before = objective_matrix.copy()
    for num in range(num_iteration):
        Q, R = np.linalg.qr(X_before)
        X_next = R @ Q
        X_before = X_next

    return X_before


def lanczos_iteration(objective_matrix, polynomial_degree):
    """
    Lanczos iteration -> find the extremum eigenvalues of symmetric matrix
    Computationally efficient due to dimension reduction of the original matrix
    Construct orthogonal vectors of Krylov subspace of polynomial degree k
    """
    # get dimension
    if objective_matrix.shape[0] != objective_matrix.shape[1]:
        raise ValueError("Insert Square Hermitian Matrix")
    matrix_dimension = objective_matrix.shape[0]

    Krylov_subspace_vector = np.zeros(shape=(matrix_dimension, polynomial_degree + 1))
    x_vector = np.random.randn(matrix_dimension)
    x_vector /= np.linalg.norm(x_vector)
    Krylov_subspace_vector[:, 0] = x_vector
    beta_list = list()
    alpha_list = list()
    for num in range(polynomial_degree):
        q = objective_matrix @ Krylov_subspace_vector[:, num]
        # gram-schmidit with q[-1]
        alpha = (Krylov_subspace_vector[:, num].reshape(1, -1) @ q.reshape(-1, 1)).item()
        q -= Krylov_subspace_vector[:, num] * alpha

        # gram-schmidit with q[-2]
        if len(beta_list) != 0:
            q -= Krylov_subspace_vector[:, num - 1] * beta_list[-1]

        # append
        alpha_list.append(alpha)
        beta = np.linalg.norm(q)
        beta_list.append(beta)
        Krylov_subspace_vector[:, num + 1] = q / beta

        # append alpha k+1
        if num == polynomial_degree - 1:
            q = objective_matrix @ Krylov_subspace_vector[:, num + 1]
            alpha = (Krylov_subspace_vector[:, num + 1].reshape(1, -1) @ q.reshape(-1, 1)).item()
            alpha_list.append(alpha)

    # construct tri-diagonal matrix T_(k+1)
    T = np.diag(alpha_list) + np.diag(beta_list, k=1) + np.diag(beta_list, k=-1)

    return T

