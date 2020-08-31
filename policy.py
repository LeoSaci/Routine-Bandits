import numpy as np


class Policy(object):
    def __init__(self):
        pass

    def __str__(self):
        """
        Return the policy name
        (type str)
        """
        pass
    
    def __color__(self):
        """
        Return the color associated to the policy
        """

    def choose(self, agent):
        """
        Choose an action and return it 
        (type int)
        """
        pass


class Random(Policy):
    def __init__(self):
        pass

    def __str__(self):
        return "Random"
    
    def __color__(self):
        return 'b'

    def choose(self, agent):
        if agent.t < K:
            a_t = int(t)
        else:
            a_t = int(np.random.choice([i for i in range(K)]))
        return a_t


class KLUCB(Policy):
    def __init__(self):
        pass
    
    def __str__(self):
        return "KLUCB"
    
    def __color__(self):
        return 'g'
    
    def choose(self, agent):
        t = agent.t
        T = agent.T
        K = agent.K
        h = agent.h

        if t < K:
            a_t = int(t % K)
        else:
            emp_means = agent.emp_mean[h]
            deviations = agent.deviation[h]
            a_t = int(np.argmax(emp_means + deviations))
        return a_t


class KLUCB_RB(Policy):
    def __init__(self):
        pass

    def __str__(self):
        return 'KLUCB-RB'
    
    def __color__(self):
        return 'r'
    
    def choose(self, agent):
        h = agent.h
        t = agent.t
        T = agent.T
        K = agent.K

        if t < K:
            a_t = int(t % K)
        elif h == 0:
            a_t = np.argmax(agent.emp_mean[h] + agent.deviation[h])
        else:

            idx = list(range(K))
            np.random.shuffle(idx)
            
            # Random arm index among the most pulled ones 
            i_max = idx[np.argmax(agent.N[h, idx])]

            # test
            positives = self.test(agent, h, i_max)
            nb_pos = len(positives)

            # agregated data
            agg_N = np.sum(agent.N[positives], axis=0) + agent.N[h]
            agg_sums = np.sum(agent.sums[positives], axis=0) + agent.sums[h]
            agg_means = agg_sums / agg_N
            y = ((nb_pos + 1) * T) / (K * agg_N)
            f_h_t = agent.log_plus(y * (agent.log_plus(y)**2 + 1))
            agg_deviation = np.sqrt((2 * f_h_t) / agg_N)
            a_t = int(np.argmax(agg_means + agg_deviation))
        return a_t

    def test(self, agent, h, i_max):
        """
        Return the list of periods k <= h-1 positively tested
        (type list)
        """
        Z = 100 * np.abs(agent.most_pulled[:h] - i_max) * np.ones((agent.K, h))
        Z = Z.T
        Z = (
            Z
            + np.abs(agent.emp_mean[:h] - agent.emp_mean[h])
            - agent.deviation[:h]
            - agent.deviation[h]
        )
        Z = np.max(Z, axis=1)
        
        positives = set(np.where(Z <= 0)[0])
        if h == (agent.H - 1):
            false_positives = positives - agent.true_positives[h]
            agent.fp_rate[int(agent.t-agent.K)] = len(false_positives) / h
            agent.tp_rate[int(agent.t-agent.K)] = len(positives - false_positives) / h
            agent.n_rate[int(agent.t-agent.K)] = (h - len(positives)) / h
        return list(positives)