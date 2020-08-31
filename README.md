## An algorithm adapted to routine bandits framework : KLUCB-RB

We propose here an Python implementation of the KLUCB-RB algorithm. MABs have arms normally distributed with means randomly generated according to the uniform distribution in [-1, 1] and variance 1.

KLUCB is also implemented for comparison and averaged cumulative regret curves resulting from both policies are plotted at the end of each run, and saved to files such as `run_x.png`.
The randomly generated bandits means are saved as a numpy matrix in a file in a compressed file `models.gz`.

#### Requirements
matplotlib, scipy, tqdm

#### Run the code
Run the script `main.py` to run a KLUCB policy and a KLUCB-RB policy and observe their cumulative regrets.
```
python main.py --n_runs --K --M --H --T --xps_name --erase
```
- Parse parameters :
--n_runs : number of runs (type int, default=1)
--K : number of arms (type int, default=2)
--M : number of bandits (type int, default=2)
--H : number of periods (type int, default=10)
--T : time horizon of each period (type int, default=1000)
--xps_name : name of the experiment (type str, default='xps1')
--save_xps : to save an experiment (type bool, default=True)
--erase : to restart an old experiment (type bool, default=False)

If --save_xps argument is True, the code creates folders /experiments and /figures in the current directory. At the end of run n, the cumulative regret matrix of the n runs is saved in ./experiments, a figure showing average regrets with one standard deviation is saved in ./figures.

When an experiment is stopped at iteration N, the code will continue the same experiment with n runs more if --n_runs argument is choosen equal to n+N and if --erase argument is False.
If --erase is False, the code will start a new experiment with new randomly generated parameters.

To keep results from several experiments, just change --xps_name before running `main.py`.