import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def sto_step(p, q, old):
    """
    Simulate one step of the stochastic process.

    Before division, allow the current sensitive cells to transition
    to resistant cells with probability p, and the current resistant
    cells to transition to sensitive cells with probability q.

    Then, double the population of both types and return the new counts.
    """
    new = np.zeros(2)

    sens_to_resist = np.random.binomial(old[0], p)
    resist_to_sens = np.random.binomial(old[1], q)

    new[0] = 2 * old[0] - 2 * sens_to_resist + 2 * resist_to_sens
    new[1] = 2 * old[1] - 2 * resist_to_sens + 2 * sens_to_resist

    return new

def sim(p, q, ics, num_gens, num_trials=10000):
    """
    Simulate 'num_gens' generations of the stochastic process 'sto_step'
    starting from an initial condition vector 'ics' = [sens_0, resist_0]
    for 'num_trials' trials.

    The data is returned as a 3D matrix with 'num_trials' layers,
    'num_gens' rows, and 2 columns (sensitive and resistant counts).

    The returned data matrix can be indexed using `data[trial#, gen#, cell_type]`
    where 'cell_type' is 0 for the sensitive and 1 for the resistant population.
    """
    data = np.zeros((num_trials, num_gens+1, 2))
    data[:, 0, :] = np.tile(ics.flatten(), (num_trials, 1))

    for trial in range(num_trials):
        for gen in range(1, num_gens+1):
            data[trial, gen, :] = sto_step(p, q, data[trial, gen-1, :])

    return data

from joblib import Parallel, delayed
def param_scan(ics, num_gens, num_trials, p_vals, q_vals, seed=42, output_dir=None, n_jobs=-1):
    """
    Given a vector of p values and q values, run the simulation for each set of parameters.

    By default, n_jobs is set to -1, which uses all of the availble CPU cores to parallelize
    the scan.

    The output includes:
    1) `settings`: a 2D array where the first row is the initial conditions vector
    and the (2,1) is `num_gens` and (2,2) is `num_trials`.
    2) `p_vals`: a vector of p values that were scanned.
    3) `q_vals`: a vector of q values that were scanned.
    4) `data`: a 5D matrix which can be indexed using `data[p_idx, q_idx, trial#, gen#, cell_type]`
    where 'cell_type' is 0 for the sensitive and 1 for the resistant population.

    To load this data into Python, use:
    ```
    import numpy as np

    output = np.load('file_path/.../simulation_output.npz')

    settings = output['settings']
    p_vals   = output["p_vals"]
    q_vals   = output["q_vals"]
    data     = output["data"]
    """
    np.random.seed(seed)

    ics = ics.flatten() # ensure correct shape

    def run_one(i, j, p, q):
        pq_run = sim(p, q, ics, num_gens, num_trials)
        return (i, j, pq_run)

    tasks = [(i, j, p, q) for i, p in enumerate(p_vals) for j, q in enumerate(q_vals)]
    results = Parallel(n_jobs=n_jobs)(delayed(run_one)(i, j, p, q) for i, j, p, q in tqdm(tasks, desc="Simulating (p, q) pairs"))

    data = np.zeros((p_vals.size, q_vals.size, num_trials, num_gens+1, 2))
    for i, j, pq_run in results:
        data[i, j, :, :, :] = pq_run

    if output_dir is not None:
        settings = np.array([ics, [num_gens, num_trials]])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"simulation_output_{timestamp}.npz")
        np.savez(filename, settings=settings, p_vals=p_vals, q_vals=q_vals, data=data)
    
    return data