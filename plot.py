import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager
import pylab
import matplotlib as mpl
from scipy import stats
import os
q = stats.norm.ppf(0.975)

mpl.rcParams['font.size'] = 22
mpl.rcParams['font.serif'] = "Computer Modern Roman"


def plot_regret(time, regret, it, fig_name, policies, H, T):
    _ = plt.figure(num=it)

    for i, pi in enumerate(policies):
        lab = pi.__str__()
        col = pi.__color__()

        regret_avg = np.mean(regret[i][:it+1], axis=0)
        regret_std = np.std(regret[i][:it+1], axis=0)

        plt.plot(time, regret_avg, label=lab, color=col)
        plt.fill_between(
            time,
            regret_avg - (q / np.sqrt(it+1)) * regret_std,
            regret_avg + (q / np.sqrt(it+1)) * regret_std,
            color=col, alpha=0.2
        )
        plt.xlabel("Period h")
        # plt.ylabel("Cumulative regret " + "$R(\\nu, h, T)$")
        plt.ylim([0, 500])
        plt.xticks(T * np.arange(H), np.arange(H)+1)
        # plt.xticks(T * 2 * np.arange(int((H+1)/2)), 2 * np.arange(int((H+1)/2)) + 1)
        plt.legend()
        plt.rc("text", usetex=True)
        pylab.rc('font', family='serif', size=15)
        plt.tight_layout()
    plt.ioff()
    if os.path.isfile(fig_name):
        os.remove(fig_name)
    plt.savefig(fig_name, frameon=False, bbox_inches="tight", pad_inches=0.01)

def plot_fp_rate(time, FPR, TPR, NEG, it, filename):
    fp_rate_avg = np.mean(FPR[:it+1], axis=0)
    fp_rate_std = np.std(FPR[:it+1], axis=0)
    tp_rate_avg = np.mean(TPR[:it+1], axis=0)
    tp_rate_std = np.std(TPR[:it+1], axis=0)
    neg_rate_avg = np.mean(NEG[:it+1], axis=0)
    neg_rate_std = np.std(NEG[:it+1], axis=0)
    _ = plt.figure(num=(1000+it))
    colors = ['b', 'r', 'k']

    plt.plot(time, fp_rate_avg, color=colors[0], label='false positives')
    plt.fill_between(
        time,
        fp_rate_avg - (q / np.sqrt(it+1)) * fp_rate_std,
        fp_rate_avg + (q / np.sqrt(it+1)) * fp_rate_std,
        color=colors[0], alpha=0.2
    )
    plt.plot(time, tp_rate_avg, color=colors[1], label='true positives')
    plt.fill_between(
        time,
        tp_rate_avg - (q / np.sqrt(it+1)) * tp_rate_std,
        tp_rate_avg + (q / np.sqrt(it+1)) * tp_rate_std,
        color=colors[1], alpha=0.2
    )
    plt.plot(time, neg_rate_avg, color=colors[2], label='negatives')
    plt.fill_between(
        time,
        neg_rate_avg - (q / np.sqrt(it+1)) * neg_rate_std,
        neg_rate_avg + (q / np.sqrt(it+1)) * neg_rate_std,
        color=colors[2], alpha=0.2
    )
    plt.xlabel('$t$')
    # plt.ylabel('False positive periods rate')
    plt.xlim([0, 500])
    plt.ylim([0, 1])
    plt.legend()
    plt.rc("text", usetex=True)
    pylab.rc('font', family='serif', size=20)
    plt.tight_layout()
    plt.savefig(filename, frameon=False, bbox_inches="tight", pad_inches=0.01)


def plot_setting(models, filename):
    # models = np.loadtxt('/home/leo/Documents/ARPE/experiments/KL-UCB++/H~25/K~4/T~20000random_parameters_exp_2/log_9/models.gz')
    # filename = '/home/leo/Documents/ARPE/final_code2/setting_exp2.pdf'
    M, K = models.shape
    fig = plt.figure()
    ax = fig.add_subplot()

    for m in range(M):
        labs = ['$b_{' + str(i + 1) + '}$' for i in range(M)]
        # labs = ['$b_{1}$', '$b_{2}$', '$b_{3}$', '$b_{4}$', '$b_{5}$']
        plt.scatter([i + 1 for i in range(K)], models[m], marker='o')
        z = np.polyfit([i + 1 for i in range(K)], models[m], K - 1)
        p = np.poly1d(z)
        plt.plot(
            np.linspace(1, K, 1000),
            p(np.linspace(1, K, 1000)),
            "--",
            label=labs[m],
        )
    for i in range(K):
        plt.plot([i + 1, i + 1], [-1, 1], "--k", linewidth=0.7)

    plt.ylim([-1.1, 1.1])
    plt.xlabel("Arm " + "$a\in\\mathcal{A}$")
    plt.xticks([i + 1 for i in range(K)])
    plt.yticks(list(np.linspace(-1, 1, 11)))
    leg = plt.legend(loc = 'upper right')
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    xOffset = 0.25
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
    # plt.savefig(filename, frameon=False, bbox_inches="tight", pad_inches=0.01)
    plt.rc("text", usetex=True)
    pylab.rc('font', family='serif', size=15)
    plt.tight_layout()
    plt.savefig(filename, frameon=False, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    # models = np.loadtxt('/home/leo/Documents/ARPE/experiments/KL-UCB++/H~25/K~4/T~20000random_parameters_exp_2/log_0/models.gz')
    # print(models)
    # filename = '/home/leo/Documents/ARPE/final_code2/setting_exp2.pdf'
    # plot_setting(models, filename)
    H = 10
    T = 10000
    from policy import KLUCB_RB, KLUCB
    policies = [KLUCB(), KLUCB_RB()]
    time = np.arange(H * T) + 1
    reg1 = np.loadtxt('/home/leo/Documents/ARPE/final_code2/experiments/M~2/K~2/H~10/T~' + str(T) + '/xp_T' + str(T) + '/KLUCB/regret.gz')
    reg2 = np.loadtxt('/home/leo/Documents/ARPE/final_code2/experiments/M~2/K~2/H~10/T~' + str(T) + '/xp_T' + str(T) + '/KLUCB-RB/regret.gz')
    regs = [reg1, reg2]
    regret = {i: regs[i] for i in range(2)}
    parameters = {'M': 2, 'K': 2, 'H': 10, 'T': T}
    savedir_fig = './figures' + ''.join(['/' + p + '~' + str(parameters[p]) for p in parameters]) + '/' + 'xp_T' + str(T)
    for it in [98, 99]:
        fig_name = savedir_fig + '/regret' + '/run_' + str(it+1) + '.pdf'
        plot_regret(time, regret, it, fig_name, policies, H, T)