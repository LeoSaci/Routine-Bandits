import numpy as np


class SwitchingAgent(object):
    def __init__(self, bandits, policy, T):
        """
        Initialize the agent for a set of bandits (type list), 
        a policy (type Policy), a Time horizon (type int)
        """
        self.bandits = bandits
        self.policy = policy
        self.T = T

    def reset(self, data, switch_seq):
        """Reset the agent"""
        self.data = data
        self.switch_seq = switch_seq
        self.H = len(data)
        self.K = data[0].shape[1]

        # matrix of shape (H, H) where the value (h, k) is 1 if b_h = b_k, 0 otherwise
        self.adjacence = np.ones((self.H, self.H))
        for h in range(self.H - 1):
            for k in range(h + 1, self.H):
                adj_hk = self.switch_seq[h] == self.switch_seq[k]
                self.adjacence[h, k] = adj_hk
                self.adjacence[k, h] = adj_hk
        self.adjacence.astype("int")

        # dictionary where a key h returns the set (type set) of periods k such that b_h = b_k
        self.true_positives = {
            h: set(list(np.where(self.adjacence[h, :h] == 1)[0]))
            for h in range(1, self.H)
        }
        self.fp_rate = np.zeros(self.T-self.K).astype('float')
        self.tp_rate = np.zeros(self.T-self.K).astype('float')
        self.n_rate = np.zeros(self.T-self.K).astype('float')

        # deviations for each period and each arm
        self.deviation = np.zeros((self.H, self.K))

        # counter of pulls for each period and each arm
        self.N = np.zeros((self.H, self.K))

        # total sums of rewards for each period and each arm
        self.sums = np.zeros((self.H, self.K))

        # empirical means for each period and each arm
        self.emp_mean = np.zeros((self.H, self.K))

        # list of choosen actions in the current period
        self.act_seq = []

        self.regret = np.zeros(self.H * self.T)
        self.FP = np.ones((self.H, self.T)) * np.arange(self.H).reshape(
            (self.H, 1)
        )
        self.h, self.t = 0, 0
        self.total_regret = 0
        self.most_pulled = np.zeros(self.H)
        self.Cl = {0: [0]}

    def run(self, data, switch_seq):
        """
        Run a policy over the H periods for 
        switching sequence switch_seq
        """
        self.reset(data, switch_seq)
        for h in range(self.H):
            self.h = h
            for t in range(self.T):
                self.t = t
                a_t = self.policy.choose(self)
                r_t = self.data[h][int(self.N[h, a_t]), a_t]
                self.update(a_t, r_t)

    def choose(self):
        """Return the arm index choosen by the policy"""
        a_t = self.policy.choose(self)
        return a_t

    def update(self, a, r):
        """Update"""
        self.act_seq.append(a)
        self.N[self.h, a] += 1
        self.sums[self.h, a] += r
        self.emp_mean[self.h, a] = self.sums[self.h, a] / self.N[self.h, a]
        self.update_deviation(a)
        if (self.t + 1) == self.T:
            i, j = self.h * self.T, (self.h + 1) * self.T
            self.regret[i:j] = self.total_regret + np.cumsum(
                self.bandits[self.switch_seq[self.h]].gaps[self.act_seq]
            )
            self.total_regret = self.regret[j - 1]
            self.act_seq = []
            self.most_pulled[self.h] = np.argmax(self.N[self.h])

    def update_deviation(self, a):
        """
        Update the deviation of arm a, and update 
        the deviations of each seen period before a switch
        """

        n_a = self.N[self.h, a]
        
        if self.policy.__str__() == "KLUCB" or self.h==0:
            y = self.T / (self.K * (n_a))
            f_t = self.log_plus(y * (self.log_plus(y) ** 2 + 1))
            self.deviation[self.h, a] = np.sqrt((2 * f_t) / n_a)

        else:
            delta = 1 / (4 * self.K * self.h * (self.t + 1) * (self.t + 2))
            self.deviation[self.h, a] = self.dev_laplace(n_a, delta)

        if (self.t + 1) == self.T:
            n = self.N[: self.h + 1]
            delta = 1 / (
                4 * self.K * (self.h + 1) * (self.t + 1) * (self.t + 2)
            )
            self.deviation[: self.h + 1] = self.dev_laplace(n, delta)

    def dev_laplace(self, n, delta):
        """Return the Laplace interval of confidence (1 - delta) after n pulls"""
        return np.sqrt(2 * ((1 + 1 / n) / n) * np.log(np.sqrt(n + 1) / delta))
    
    def log_plus(self, x):
        if type(x) == np.ndarray:
            s = x.shape[0]
            return np.log(x) * (np.log(x) > 0)
        else:
            return np.max([np.log(x), 0])
