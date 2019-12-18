# Sample program for QAOA with cirq

# IMPORTS
import sys
import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
import cirq


# helper function for constructing Ising matrix
def get_ising():
    """
    function to construct Ising matrix
    """
    # define problem on graph
    n = 5
    # Ising (interaction) matrix
    ising = np.zeros((n, n))
    ising[0, 1] = 1
    ising[0, 4] = 1
    ising[1, 2] = 1
    ising[1, 3] = 1
    ising[2, 3] = 1
    ising[3, 4] = 1
    # print Ising matrix (wikipedia example)
    print('Ising matrix:\n', ising)
    return ising, n


# helper function for evolution with mixer Hamiltonian
def mixer(beta):
    """
    Generator for U(beta, B) layer (mixing layer) of QAOA
    """
    for qubit in qubits:
        yield cirq.X(qubit)**beta


# helper function for evolution with mixer Hamiltonian
def cost_circuit(gamma):
    """
    returns circuit for evolution with cost Hamiltonian
    """
    for ii in range(N):
        for jj in range(ii+1, N):
            yield cirq.ZZ(qubits[ii], qubits[jj])**(gamma*J[ii, jj])


# function to build the QAOA circuit with depth p
def circuit(params):
    """
    function to return full QAOA circuit
    """

    # initialize qaoa circuit with first Hadamard layer
    circ = cirq.Circuit()
    circ.append(cirq.X.on_each(*[q for q in qubits]))  # for minimization start in |->
    circ.append(cirq.H.on_each(*[q for q in qubits]))

    # setup two parameter families
    circuit_length = int(len(params) / 2)
    gammas = params[:circuit_length]
    betas = params[circuit_length:]

    # add circuit layers
    for mm in range(circuit_length):
        circ.append(cost_circuit(gammas[mm]))
        circ.append(mixer(betas[mm]))

    return circ


# helper function to compute energy for given measurement result
def get_energy(meas):
    """
    function to get energy estimate for given measurement results
    """
    # covert 0 bit values to -1 ising
    meas[meas == 0] = -1

    # TODO: vectorize this expression
    energy = 0
    for ii in range(N):
        for jj in range(ii + 1, N):
            energy += J[ii, jj] * meas[ii] * meas[jj]

    return energy


# function that computes cost function for given params
def objective_function(params):
    """
    objective function takes a list of variational parameters as input,
    and returns the cost associated with those parameters
    """
    # obtain a quantum circuit instance from the parameters
    qaoa_circuit = circuit(params)
    # add final measurements in z basis
    measurement = cirq.measure(*qubits, key='z')
    qaoa_circuit.append(measurement)
    # print the circuit w/ measurement layer
    # print('Print quantum circuit:')
    # print(qaoa_circuit)

    # classically simulate the circuit
    # run quantum circuit to obtain the probability distribution associated with the current parameters
    simulator = cirq.Simulator()
    result = simulator.run(qaoa_circuit, repetitions=NUM_SHOTS)

    # TODO: keep track of optimal classical result and corresponding bitstring
    # reformat classical results and compute cost
    hist = result.histogram(key='z')
    probs = [v / result.repetitions for _, v in hist.most_common(NUM_SHOTS)]
    configs = [c for c, _ in hist.most_common(NUM_SHOTS)]
    meas = [[int(s) for s in ''.join([str(b) for b in bin(k)[2:]]).zfill(N)] for k in configs]
    costs = [get_energy(np.array(m)) for m in meas]
    print('Costs of individual shots:', costs)
    energy_expect = np.dot(probs, costs)
    print('Approx energy expectation value:', energy_expect.round(5))
    # NOTE: could consider other definitions of cost function
    cost = energy_expect

    return cost


# helper function for plotting optimal angles
def plot_angles(optimal_angles, depth):
    """
    function to plot optimal angles as found by QAOA
    """

    # plot optimal angles
    angles_last_run = optimal_angles[-1]
    gamma = angles_last_run[:depth]
    beta = angles_last_run[depth:]
    pa = np.arange(1, depth + 1)

    fig = plt.figure(1)
    plt.plot(pa, gamma / np.pi, '-o', label='gamma')
    plt.plot(pa, beta / np.pi, '-s', label='beta')
    plt.xlabel('circuit depth p')
    plt.ylabel('optimal angles [pi] (global optim)')
    plt.legend(title='Variational QAOA angles:', loc='lower left')
    plt.show()


# The function to execute the training: run classical minimization.
def train(options, p=6, n_initial=5):
    """
    function to run QAOA algorithm for given, fixed circuit depth p
    """
    print('Starting the training.')

    # initialize vectors to store minimum energy and optimal angles
    energies = []
    optim_angles = []

    print('==================================' * 3)
    print('OPTIMIZATION for circuit depth p={depth}'.format(depth=p))

    # initialize
    cost_energy = []
    angles = []
    # TODO: remove for loop over initial random seeds (different jobs)
    # optimize for different random initializations (double check value range)
    n_initial = max(n_initial, 2 ** (p + 1))
    for ii in range(n_initial):
        # randomly initialize variational parameters
        # angles0 = np.random.normal(size=2 * p)
        # randomly initialize variational parameters within appropriate bounds
        gamma_initial = np.random.uniform(0, 2 * np.pi, p).tolist()
        beta_initial = np.random.uniform(0, np.pi, p).tolist()
        params0 = np.array(gamma_initial + beta_initial)
        params0[0] = 0.01
        params0[1] = 0.4
        # set bounds for search space
        # TODO: double check parameter bounds (pi factors)
        # bnds = [(0, np.pi / 2) for _ in range(len(angles0))]
        bnds_gamma = [(0, 2 * np.pi) for _ in range(int(len(params0) / 2))]
        bnds_beta = [(0, np.pi) for _ in range(int(len(params0) / 2))]
        bnds = bnds_gamma + bnds_beta
        # run classical optimization
        # TODO: use callbacks
        # note: support callbacks https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        # result = minimize(objective_function, params0, options=options, method='SLSQP', bounds=bnds)
        result = minimize(objective_function, params0, options=options, method='Nelder-Mead')
        # store result of classical optimization
        result_energy = result.fun
        cost_energy.append(result_energy)
        print('Optimal energy (single run):', result_energy)
        result_angle = result.x
        angles.append(result_angle)
        print('Optimal angles (single run):', result_angle)
    # store energy minimum (over different initial configurations)
    energy_min = np.min(cost_energy)
    energies.append(energy_min)
    optim_angles.append(angles[np.argmin(cost_energy)])
    print('==================================' * 3)
    print('Optimization done for circuit depth {p} with minimal energy {E}'.format(p=p, E=energy_min))

    # plot optimal angles
    plot_angles(optim_angles, p)


if __name__ == '__main__':

    # set up the problem
    NUM_SHOTS = 1000
    J, N = get_ising()
    p = 5  # circuit depth
    n_initial = 5  # initial random seeds
    # set options for classical optimization
    # options = {'disp': True}
    options = {}

    # define qubits and quantum circuit
    qubits = [cirq.LineQubit(x) for x in range(N)]

    # kick off training
    # TODO: make use of Z2 symmetry
    start = time.time()
    train(options=options, p=p, n_initial=n_initial)
    end = time.time()
    # print execution time
    print('Code execution time [sec]:', end - start)

    # Zero exit code causes the job to be marked a Succeeded
    sys.exit(0)
