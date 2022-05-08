import numpy as np

from math import factorial, ceil
from numpy.polynomial.hermite import Hermite, hermgauss


def hermite_function(n):
    degree_vector = [0] * n + [1]
    hermite_polynomial = Hermite(coef=degree_vector)
    factor = (np.sqrt(np.pi) * 2 ** n * factorial(n)) ** (-0.5)  # for normalization

    def hermite_func(x):
        return hermite_polynomial(x) * np.exp(-0.5 * x ** 2) * factor

    return hermite_func


def get_kinetic_wave_function(n, h=1, m=1):
    # pre-factor
    kinetic_factor = -0.5 * h ** 2 / m
    degree_vector = [0] * n + [1]

    hermite_polynomial = Hermite(coef=degree_vector)
    factor = (np.sqrt(np.pi) * 2 ** n * factorial(n)) ** (-0.5)  # for normalization

    first_derivative = hermite_polynomial.deriv(m=1)
    second_derivative = hermite_polynomial.deriv(m=2)

    def return_func(x):
        result = hermite_polynomial(x) * factor * (x ** 2 - 1) * np.exp(-0.5 * x ** 2) \
                 - 2 * factor * first_derivative(x) * x * np.exp(-0.5 * x ** 2) + second_derivative(x) * np.exp(
            -0.5 * x ** 2) * factor
        result *= kinetic_factor

        return result

    return return_func


def get_potential_wave_function(n, m=1, angular_w=1):
    # pre-factor
    potential_factor = 0.5 * m * angular_w ** 2
    degree_vector = [0] * n + [1]

    hermite_polynomial = Hermite(coef=degree_vector)
    factor = (np.sqrt(np.pi) * 2 ** n * factorial(n)) ** (-0.5)  # for normalization

    def return_func(x):
        return hermite_polynomial(x) * potential_factor * factor * x ** 2 * np.exp(-0.5 * x ** 2)

    return return_func


def multiply_two_functions(func1, func2):
    def return_function(x):
        return func1(x) * func2(x)

    return return_function


def calculate_integral(func, degree):
    nodes, weights = hermgauss(deg=degree)
    summation = 0
    for node, weight in zip(nodes, weights):
        summation += func(node) * np.exp(node ** 2) * weight
    return summation


def get_hamiltonian_ij(func_i_degree, func_j_degree, h=1, m=1, angular_w=1):
    # get function
    func_i = hermite_function(func_i_degree)
    kinetic_func_j = get_kinetic_wave_function(n=func_j_degree, h=h, m=m)
    potential_func_j = get_potential_wave_function(n=func_j_degree, m=m, angular_w=angular_w)

    # calculate integral
    required_degree = ceil((func_i_degree + func_j_degree + 2 + 1) / 2)
    kinetic_integral = calculate_integral(
        func=multiply_two_functions(func1=func_i, func2=kinetic_func_j), degree=required_degree
    )
    potential_integral = calculate_integral(
        func=multiply_two_functions(func1=func_i, func2=potential_func_j), degree=required_degree
    )

    return kinetic_integral + potential_integral

