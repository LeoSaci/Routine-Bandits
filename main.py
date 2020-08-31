import argparse
import os
import numpy as np
from tqdm import tqdm
from bandit import GaussianBandit
from data import build_dataset
from agent import SwitchingAgent
from policy import KLUCB_RB, KLUCB
from plot import plot_regret, plot_setting, plot_fp_rate

# Parameters
parser = argparse.ArgumentParser(description='Experiments KLUCB-RB')
parser.add_argument('--n_runs', type=int, default=1, help="number of runs for the experiment")
parser.add_argument('--K', type=int, default=2, help="number of arms")
parser.add_argument('--M', type=int, default=2, help="number of bandits")
parser.add_argument('--H', type=int, default=10, help="number of periods")
parser.add_argument('--T', type=int, default=1000, help="time horizon")
parser.add_argument('--xps_name', type=str, default='xps1', help="name of the experiment")
parser.add_argument('--save_xps', type=bool, default=True, help="to save the experiments if True")
parser.add_argument('--erase', type=bool, default=False, help="To continue a previous experiment if False, or restart the experiment if True")
args = parser.parse_args()


n_runs = args.n_runs
K = args.K
M = args.M
T = args.T
H = args.H
xps_name = args.xps_name
save_xps = args.save_xps
erase = args.erase
root_path_exp = './experiments'
root_path_fig = './figures'
parameters = {'M': M, 'K': K, 'H': H, 'T': T}

# policies
policies = [KLUCB(), KLUCB_RB()]

print("Number of models : M = " + str(M))
print("Number of arms : K = " + str(K))
print("Number of periods : H = " + str(H))
print("Time horizon : T = " + str(T))
print("Number of runs : " + str(n_runs))
print()

# Arms means for each bandit; numpy array, shape : (M, K)
models = 2 * np.random.rand(M, K) - 1


regret = {i: np.zeros((n_runs, H * T)) for i in range(len(policies))}
time = np.arange(H * T) + 1
savedir_xps = root_path_exp + ''.join(['/' + p + '~' + str(parameters[p]) for p in parameters]) + '/' + xps_name
savedir_fig = root_path_fig + ''.join(['/' + p + '~' + str(parameters[p]) for p in parameters]) + '/' + xps_name
paths_xps = [savedir_xps + '/' + pi.__str__() for pi in policies]

for i in range(len(paths_xps)):
    if not os.path.exists(paths_xps[i]):
        os.makedirs(paths_xps[i])
if not os.path.exists(savedir_fig + '/regret'):
    os.makedirs(savedir_fig + '/regret')
if not os.path.exists(savedir_fig + '/fp_rate'):
    os.makedirs(savedir_fig + '/fp_rate')

if (not erase) and os.path.isfile(savedir_xps + '/models.gz'):
    models = np.loadtxt(savedir_xps + '/models.gz')
    start_it = n_runs + 1
    for i, pi in enumerate(policies):
        filename = paths_xps[i] + '/regret.gz'
        if os.path.isfile(filename):
            reg_i = np.loadtxt(filename)
            temp = np.min((reg_i.shape[0], n_runs))
            regret[i][:temp, :] = reg_i[:temp, :]
        else:
            temp = 0
        start_it = np.min((start_it, temp))
    if start_it == (n_runs + 1):
        start_it = 0
    if start_it == 0:
        regret = {i: np.zeros((n_runs, H * T)) for i in range(len(policies))}
else:
    start_it = 0


# switch_sequences = np.random.randint(0, M, (n_runs-start_it, H)).astype('int').reshape((n_runs-start_it, H))
switch_sequences = np.array([[(i+j)%M for j in range(H)] for i in range(n_runs)])
# switch_sequences = np.zeros((n_runs, H)).astype('int')
bandits = [GaussianBandit(mu) for mu in models]
datasets = [build_dataset(bandits, T, H, seq) for seq in switch_sequences]
agent = {i: SwitchingAgent(bandits, pi, T) for i, pi in enumerate(policies)}
FPR = np.zeros((n_runs, T-K))
TPR = np.zeros((n_runs, T-K))
NEG = np.zeros((n_runs, T-K))

for it in tqdm(range(start_it, n_runs)):
    data = datasets[it-start_it]
    switch_seq = switch_sequences[it-start_it]
    for i, pi in enumerate(policies):
        agent[i].run(data, switch_seq)
        regret[i][it] = agent[i].regret
        if pi.__str__() == 'KLUCB-RB':
            FPR[it] = agent[i].fp_rate
            TPR[it] = agent[i].tp_rate
            NEG[it] = agent[i].n_rate
        if save_xps:
            if it == 1  and i == 0:
                np.savetxt(savedir_xps + '/models.gz', models)
                plot_setting(models=models, filename=savedir_fig + '/' + xps_name + '.pdf')
            np.savetxt(paths_xps[i] + '/regret.gz', regret[i][:it+1])
            path_i_xps = paths_xps[i] + '/log_' + str(it+1)
            fig_name = savedir_fig + '/regret' + '/run_' + str(it+1) + '.pdf'
            fig_name_fp = savedir_fig + '/fp_rate' + '/fp_run_' + str(it+1) + '.pdf'
    if save_xps:
        plot_regret(time, regret, it, fig_name, policies, H, T)
        plot_fp_rate(np.arange(T-K), FPR, TPR, NEG, it, fig_name_fp)