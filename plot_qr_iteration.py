import numpy as np
import matplotlib.pyplot as plt

from iteration import inverse_power_iteration, qr_iteration, lanczos_iteration

if __name__ == '__main__':
    # load hamiltonian
    hamiltonian = np.load('hamiltonian_size_20.npy')

    # similarity transformation
    np.random.seed(42)
    n = hamiltonian.shape[0]
    H = np.random.randn(n, n)
    Q, R = np.linalg.qr(H)
    modified_hamiltonian = Q @ hamiltonian @ Q.T

    # comparison
    num_iteration = 200

    # QR iteration -> get eigenvalue & eigenvector pairs
    energy_idx = [0, 2, 4, 6, 8, 10, 12]
    eigen_value_matrix = qr_iteration(objective_matrix=modified_hamiltonian, num_iteration=num_iteration)
    qr_energy = np.flip(eigen_value_matrix.diagonal())
    true_energy = hamiltonian.diagonal()
    error = np.abs(qr_energy - true_energy) / true_energy

    # plot
    # energy_tick = ['E%d' % num for num in energy_idx]
    plt.figure(figsize=(8, 8), dpi=300)
    plt.title("Energy of quantum harmonic oscillator (matrix size = 20)")
    plt.plot(true_energy[energy_idx], error[energy_idx], 'ro', label='QR iteration')
    plt.xticks(true_energy[energy_idx])
    plt.hlines(y=1e-16, xmin=true_energy[0], xmax=true_energy[energy_idx[-1]], color='black', linestyles='-', label='Machine precision ($10^{-16}$)')
    for idx in energy_idx:
        plt.vlines(x=true_energy[idx], ymax=error[idx], ymin=1e-16, color='grey', linestyles='--', alpha=0.5)
    plt.yscale('log')
    plt.xlabel('True energy (unit: $\hbar$)', fontsize=10)
    plt.ylabel('Relative Error', fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig('qr_test.png')
