"""Compare a classical exhaustive portfolio search with a Qiskit QAOA simulation."""

import time

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
import scipy.stats


def score(portfolio, returns, covariance):
    """Return the portfolio objective: expected return minus risk."""
    if portfolio.sum() == 0:
        # empty portfolio is not valid for this objective.
        return -float("inf")

    # mean-variance objective: maximize expected return while minimising risk
    return float(portfolio @ (returns - 2 * covariance @ portfolio))


def brute_force(returns, covariance):
    """Try every portfolio and keep the best one."""
    asset_count = len(returns)
    portfolios = portfolio_states(asset_count)
    portfolio_scores = portfolio_score_table(portfolios, returns, covariance)

    # skip the all-zero portfolio, very trivial here
    best_number = int(np.argmax(portfolio_scores[1:])) + 1
    best_portfolio = portfolios[best_number]
    best_score = portfolio_scores[best_number]

    return best_portfolio, best_score


def timed_classical_search(returns, covariance):
    """Run the original exhaustive-search loop used for the graph."""
    asset_count = len(returns)
    best_score = -float("inf")

    for number in range(1, 2**asset_count):
        portfolio = np.array([(number >> bit) & 1 for bit in range(asset_count)])
        portfolio_score = score(portfolio, returns, covariance)
        if portfolio_score > best_score:
            best_score = portfolio_score


def portfolio_states(asset_count):
    """Return every binary portfolio state for the requested asset count."""
    numbers = np.arange(2**asset_count)[:, None]
    bits = np.arange(asset_count)
    return ((numbers >> bits) & 1).astype(int)


def portfolio_score_table(portfolios, returns, covariance):
    """Score all portfolio states at once for faster benchmarking."""
    return portfolios @ returns - 2 * np.einsum(
        "bi,ij,bj->b", portfolios, covariance, portfolios
    )


def qaoa_probabilities(returns, covariance, gamma, beta):
    """Build and simulate a single QAOA layer with the fastest local Qiskit backend."""
    asset_count = len(returns)
    circuit = QuantumCircuit(asset_count)

    # prepare the quantum superposition over all binary portfolio choices
    circuit.h(range(asset_count))

    # use the same return and risk objective as the classical search
    # encode the objective as a diagonal phase operator with the return/risk terms
    diagonal = returns - 2 * np.diag(covariance)
    for asset in range(asset_count):
        circuit.rz(-gamma * diagonal[asset], asset)

    for first in range(asset_count):
        for second in range(first + 1, asset_count):
            circuit.rzz(-2 * gamma * covariance[first, second], first, second)

    for asset in range(asset_count):
        circuit.rx(2 * beta, asset)

    circuit.save_statevector()
    simulator = AerSimulator(method="statevector")
    statevector = simulator.run(circuit).result().get_statevector()
    return np.abs(statevector.data) ** 2


def qiskit_sampling(returns, covariance):
    """Sample one QAOA statevector and return the most likely portfolio."""
    portfolios = portfolio_states(len(returns))
    portfolio_scores = np.nan_to_num(
        portfolio_score_table(portfolios, returns, covariance), nan=0.0, posinf=0.0, neginf=0.0
    )

    probabilities = qaoa_probabilities(returns, covariance, 0.5, 0.5)
    samples = np.random.default_rng().choice(
        len(probabilities), size=1024, p=probabilities
    )
    best_number = int(np.bincount(samples).argmax())
    portfolio = portfolios[best_number]
    return portfolio, float(portfolio_scores[best_number])


def qaoa(returns, covariance):
    """Run the Qiskit QAOA simulation and return the sampled portfolio."""
    asset_count = len(returns)
    portfolios = portfolio_states(asset_count)
    portfolio_scores = np.nan_to_num(
        portfolio_score_table(portfolios, returns, covariance), nan=0.0, posinf=0.0, neginf=0.0
    )
    portfolio_scores[0] = 0.0

    # optimise the expected classical score over the quantum probabilities
    def expected_score(parameters):
        probabilities = qaoa_probabilities(returns, covariance, *parameters)
        return -float(probabilities @ portfolio_scores)

    result = minimize(
        expected_score,
        [0.5, 0.5],
        method="Nelder-Mead",
        options={"maxiter": 12},
    )
    probabilities = qaoa_probabilities(returns, covariance, *result.x)
    probabilities[0] = -1.0
    best_number = int(np.argmax(probabilities))
    best_portfolio = portfolios[best_number]
    return best_portfolio, float(portfolio_scores[best_number])


def load_market_data(tickers, period="2y"):
    """Download historical prices and convert them to daily returns."""
    prices = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )["Close"]
    prices = prices.dropna(axis="columns", how="all").dropna()

    if prices.empty or prices.shape[1] != len(tickers):
        raise RuntimeError("Could not download complete price data for all tickers")

    returns = prices.pct_change().dropna()
    return returns.mean().to_numpy(), returns.cov().to_numpy()


def benchmark(returns, covariance):
    """Time both methods through 12 assets and save the search-time graph."""
    # evaluate a range of portfolio sizes to compare scaling behaviour visually
    asset_counts = [count for count in range(2, 13, 2) if count <= len(returns)]
    classical_times = []
    qiskit_times = []

    for asset_count in asset_counts:
        # benchmark both approaches on the same subset of assets
        subset_returns = returns[:asset_count]
        subset_covariance = covariance[:asset_count, :asset_count]

        start = time.perf_counter()
        timed_classical_search(subset_returns, subset_covariance)
        classical_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        qaoa(subset_returns, subset_covariance)
        qiskit_times.append(time.perf_counter() - start)

    max_time = max(max(classical_times), max(qiskit_times)) if classical_times and qiskit_times else 0.0
    y_upper = max(0.05, max_time * 1.2) if max_time > 0 else 0.05

    plt.figure(figsize=(13, 7.65))
    plt.plot(
        asset_counts,
        classical_times,
        "o-",
        linewidth=2,
        markersize=8,
        label="Classical search",
    )
    plt.plot(
        asset_counts,
        qiskit_times,
        "o-",
        linewidth=2,
        markersize=8,
        label="Qiskit sampling",
    )
    plt.xlabel("Number of assets", fontsize=16)
    plt.ylabel("Measured time (seconds)", fontsize=16)
    plt.title("Classical vs QAOA runtime comparison", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, y_upper)
    plt.legend(loc="upper left", fontsize=15)
    plt.grid(True)
    plt.subplots_adjust(left=0.108, right=0.942, bottom=0.113, top=0.925)
    plt.savefig("benchmark_runtime_graph.png")
    plt.close()


def main():
    # assets used for the observed comparison runs
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM",
        "META", "TSLA", "AVGO", "V", "MA", "UNH",
    ]
    returns, covariance = load_market_data(tickers)

    # compare the exact classical result vs the quantum opt. results
    print("Running both methods side by side: classical exhaustive search and QAOA simulation.")
    benchmark(returns, covariance)
    classical_portfolio, classical_score = brute_force(returns, covariance)
    quantum_portfolio, quantum_score = qaoa(returns, covariance)

    print("\nClassical portfolio:", np.nonzero(classical_portfolio)[0].tolist())
    print("Classical score:", round(classical_score, 4))
    print("QAOA portfolio:", np.nonzero(quantum_portfolio)[0].tolist())
    print("QAOA score:", round(quantum_score, 4))
    print("\nComparison: the classical method is the exact benchmark for this objective.")
    print("The QAOA result is an approximate simulation that can be visually compared against it.")
    print("No automatic routing decision is made; both methods are run and compared.")
    print("Saved graph: benchmark_runtime_graph.png")


if __name__ == "__main__":
    main()
