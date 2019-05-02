import numpy as np
import pandas as pd

class Simulation:
    ''' Simulate a process of randomly selecting one of N coins, flipping the
    selected coin a certain number of times, and then repeating this a few times
    '''

    def __init__(self,
                 n_sequences=10,
                 n_reps_per_sequence=7,
                 p=(0.1, 0.8),
                 seed=0):
        np.random.seed(seed)
        self.n_sequences = n_sequences
        self.n_reps_per_sequence = n_reps_per_sequence
        self.p = p
        self.n_coins = len(p)

    def choose_coin(self):
        return np.random.choice(range(self.n_coins), 1)[0]

    def one_sequence(self):
        which_coin = self.choose_coin()
        prob = self.p[which_coin]
        return {
            'values': np.random.binomial(1, prob, size=self.n_reps_per_sequence),
            'true_p': prob
        }

    def run(self):
        data = [self.one_sequence() for k in range(self.n_sequences)]
        df = pd.DataFrame(np.array([d['values'] for d in data]))
        df.columns = ['flip_' + str(k) for k in range(self.n_reps_per_sequence)]
        df['true_p'] = [d['true_p'] for d in data]
        return df


class EM:
    '''
    Problem: Given a sequence of sequences of coin flips and given integer K>1, model the data as
    a multinomial choice of coin followed by a sequence of flips for the chosen coin.

    Thus we are estimating two length-K vectors of parameters; the bernoulli P(heads) for each
    coin and the multinomial probabilities of drawing each coin.

    z_i is a length-n_coin vector of one-hot assignment to a coin

    Proceed iteratively by
    -- Fix Q_ik = p(z_ik | x_i, theta) = p(z_ik, x_i | theta) / p(x_i | theta)
    -- Fix theta as the optimizer of sum_i sum_ik Q_ik log p(z_ik, x_i | theta)
    '''
    def __init__(self, X, n_coins=2, p_diff_tol=0.0001, max_iter=100):
        self.seq_n = X.shape[1]
        self.X_full = X
        # Assuming independence between flips within-sequence, we only need the simpler sufficient stats:
        self.X = X.sum(axis=1)
        self.Xc = X.shape[1] - self.X
        self.K = n_coins
        self.p_diff_tol = p_diff_tol
        self.max_iter = max_iter

    def _as_vec_theta(self, theta):
        return np.concatenate((theta['bernoulli_p'], theta['multinomial_p']))

    def _as_dict_theta(self, theta):
        return {
            'bernoulli_p': theta[:self.K],
            'multinomial_p': theta[self.K:]
        }

    def random_start(self):
        self.theta = {
            'bernoulli_p': np.random.uniform(size=self.K),
            'multinomial_p': np.ones(self.K) / self.K
        }
        self.theta_vec = self._as_vec_theta(self.theta)

    def _lph(self, p):
        ''' Log probability of the observed heads for a given p '''
        return self.X * np.log(p)

    def _lpt(self, p):
        ''' Log probability of the observed tails for a given p '''
        return self.Xc * np.log(1-p)

    def log_pX_given_z(self):
        ''' Log p(x | z, theta) '''
        return np.array([
            self._lph(p) + self._lpt(p) for p in self.theta['bernoulli_p']
            ]).transpose()

    def log_pX_and_z(self, logPx_given_z):
        ''' Log p(x,z | theta)
        '''
        log_p_z = np.ones((len(self.X), len(self.theta['multinomial_p'])))
        log_p_z *= np.log(self.theta['multinomial_p'])
        return logPx_given_z + log_p_z

    def E_step(self):
        ''' Compute the matrix Q_ik with nrow(X) rows and K columns '''
        logPx_given_z = self.log_pX_given_z()
        logPxz = self.log_pX_and_z(logPx_given_z)
        # Sum over levels of z to get p(x) marginally:
        logPx = np.log(np.exp(logPxz).sum(axis=1))
        logPx_mat = np.repeat(logPx.reshape((len(logPx), 1)), self.K, axis = 1)
        self.Q = np.exp(logPxz - logPx_mat)

    def maximize_step(self):
        ''' Select the maximum-likelihood assignments assuming self.p is correct

        In general this requires solving the system of first-order conditions, but here
        I've just kind of guessed at the formula for the optimum
        '''
        xp = self.X/self.seq_n
        p_hat = (xp @ self.Q)/self.Q.sum(axis=0)
        q_hat = self.Q.sum(axis=0)/self.Q.sum()
        return np.concatenate((p_hat, q_hat))

    def run(self):
        self.random_start()
        iter = 0
        step_size = np.inf
        while (step_size > self.p_diff_tol) and (iter < self.max_iter):
            iter += 1
            self.E_step()
            new_theta_vec = self.maximize_step()
            diff = self.theta_vec - new_theta_vec
            step_size = np.sqrt(np.sum(diff ** 2))
            self.theta_vec = new_theta_vec
            self.theta = self._as_dict_theta(new_theta_vec)
        self.iter=iter

    def fitted_bernoulli_p(self, mode=False):
        '''
        Based on the fitted model, return the estimated bernoulli P(heads) behind each
        data sequence.


        :param mode:
            - Default `False` means compute this as sum_z p_z_hat P(z |x, theta), the
            probability of heads, averaged over the coins according to the posterior distribution
            over coins for each sequence.
            - If `True`, instead return simply p_z_hat, the probability of heads for coin
            that maximizing P(z |x, theta)
        '''
        if mode:
            return [self.theta['bernoulli_p'][q.argmax()] for q in self.Q]
        else:
            return self.Q @ self.theta['bernoulli_p']

