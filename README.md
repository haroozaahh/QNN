# Comparing Quantum and Classical Methods of Portfolio Optimisation

a small prototype that downloads market data and runs both of these methods side by side:

- **classical brute search** for the exact benchmark.
- **QAOA simulation** using a mean-variance QUBO.

the objective is to maximise the equation $C(x)=\mu^Tx-\lambda x^T\Sigma x$, where each binary variable
selects an asset. the classical result is the exact benchmark for the same
objective. the script measures runtime and compares the resulting portfolios and
scores visually, and we can visually analyse the modern day comparison of quantum                                  and classical methods for portfolio analysis.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python portfolio_quantum_router.py
```

the run prints both selected portfolios and compares 2, 4, ..., 12 assets using
real Yahoo finance data, creating:

- `benchmark_runtime_graph.png`

## clarification

the graph contains measured wall-clock times from this computer. it is an
implementation comparison, not a claim of quantum advantage. QAOA is simulated
with qiskit's statevector backend, so this is not a hardware execution, it is still very much all 
ran on a classical computer.the script is designed to show the classical and quantum outputs in conjunction to one another, 
with the graph representing a clear visual aid of how the run time compares to one another.


this project isnt aimed to demonstrate a quantum advantage, if anything,
demonstrates there is a long way to progress for quantum ML . one could
run a similar algorithm on IBM's quantum computer simulator for larger
assets, as when attempted here, the graph tends to crash , generate
a different image each time.




