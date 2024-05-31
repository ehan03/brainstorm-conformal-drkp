# standard library imports
import itertools
import warnings
from typing import Tuple

# third party imports
import cvxpy as cp
import numpy as np

# local imports


class NaiveKelly:
    """
    Kelly bet sizing strategy for multiple bouts including a risk free asset
    """

    def __init__(
        self,
        red_probs: np.ndarray,
        blue_probs: np.ndarray,
        red_odds: np.ndarray,  # Decimal odds
        blue_odds: np.ndarray,  # Decimal odds
        current_bankroll: float,
        fraction: float = 0.10,
        min_bet: float = 0.10,
    ):
        """
        Initialize the NaiveKelly object
        """

        self.red_probs = red_probs
        self.blue_probs = blue_probs
        self.red_odds = red_odds
        self.blue_odds = blue_odds
        self.current_bankroll = current_bankroll
        self.fraction = fraction  # Default is 1/10
        self.min_bet = min_bet  # DraftKings requires a minimum $0.10 bet

        self.n = len(red_probs)
        self.variations = np.array(list(itertools.product([1, 0], repeat=self.n)))

        self.no_bet = np.identity(2 * self.n + 1)[-1]

    def __create_returns_matrix(self) -> np.ndarray:
        """
        Create returns matrix R
        """

        R = np.zeros(shape=(2 * self.n + 1, self.variations.shape[0]))
        R[-1, :] = 1
        for j in range(self.n):
            R[2 * j, :] = np.where(self.variations[:, j] == 1, self.red_odds[j], 0)
            R[2 * j + 1, :] = np.where(self.variations[:, j] == 0, self.blue_odds[j], 0)

        return R

    def __create_probabilities_vector(self) -> np.ndarray:
        """
        Create probabilities vector pi_hat, contains probability combinations
        for all possible overall event outcomes
        """

        pi_hat = np.ones(self.variations.shape[0])
        for j in range(self.n):
            pi_hat = np.where(
                self.variations[:, j] == 1,
                pi_hat * self.red_probs[j],
                pi_hat * self.blue_probs[j],
            )

        return pi_hat

    def __calculate_optimal_wagers(self) -> np.ndarray:
        """
        Calculate optimal fractions
        """

        pi_hat = self.__create_probabilities_vector()
        R = self.__create_returns_matrix()
        b = cp.Variable(2 * self.n + 1)

        objective = cp.Maximize(pi_hat @ cp.log(R.T @ b))
        constraints = [
            cp.sum(b) == 1,
            b >= 0,
        ]
        problem = cp.Problem(objective, constraints)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem.solve(solver=cp.CLARABEL)
                return b.value
        except:
            return self.no_bet

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate optimal wager amounts in dollars
        """

        fractions = self.__calculate_optimal_wagers()
        wagers = self.fraction * self.current_bankroll * fractions[:-1]
        wagers_rounded = np.round(wagers, 2)
        wagers_clipped = np.where(wagers_rounded < self.min_bet, 0, wagers_rounded)

        red_wagers, blue_wagers = wagers_clipped[::2], wagers_clipped[1::2]

        return red_wagers, blue_wagers


class DistributionalRobustKelly:
    def __init__(
        self,
        p0_p1: np.ndarray,
        red_odds: np.ndarray,  # Decimal odds
        blue_odds: np.ndarray,  # Decimal odds
        current_bankroll: float,
        fraction: float = 0.10,
        min_bet: float = 0.10,
    ):
        """
        Initialize the DistributionalRobustKelly object
        """

        self.p0_p1 = p0_p1
        self.red_odds = red_odds
        self.blue_odds = blue_odds
        self.current_bankroll = current_bankroll
        self.fraction = fraction  # Default is 1/10
        self.min_bet = min_bet  # DraftKings requires a minimum $0.10 bet

        self.n = len(p0_p1)
        self.variations = np.array(list(itertools.product([1, 0], repeat=self.n)))

        self.no_bet = np.identity(2 * self.n + 1)[-1]

    def __create_returns_matrix(self) -> np.ndarray:
        """
        Create returns matrix R
        """

        R = np.zeros(shape=(2 * self.n + 1, self.variations.shape[0]))
        R[-1, :] = 1
        for j in range(self.n):
            R[2 * j, :] = np.where(self.variations[:, j] == 1, self.red_odds[j], 0)
            R[2 * j + 1, :] = np.where(self.variations[:, j] == 0, self.blue_odds[j], 0)

        return R

    def __get_inequality_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create matrix A and vector c for the linear inequality constraint A*pi <= c
        """

        id_ = np.identity(self.variations.shape[0])
        A = np.vstack((-id_, id_))

        pi_l = np.ones(self.variations.shape[0])
        for j in range(self.n):
            pi_l = np.where(
                self.variations[:, j] == 1,
                pi_l * self.p0_p1[j, 0],
                pi_l * (1 - self.p0_p1[j, 1]),
            )

        pi_h = np.ones(self.variations.shape[0])
        for j in range(self.n):
            pi_h = np.where(
                self.variations[:, j] == 1,
                pi_h * self.p0_p1[j, 1],
                pi_h * (1 - self.p0_p1[j, 0]),
            )

        c = np.concatenate((-pi_l, pi_h))

        return A, c

    def __calculate_optimal_wagers(self) -> np.ndarray:
        """
        Calculate optimal fractions
        """

        R = self.__create_returns_matrix()
        b = cp.Variable(2 * self.n + 1)
        lmbda = cp.Variable(2 * self.variations.shape[0])
        A, c = self.__get_inequality_constraints()
        wc_growth_rate = cp.min(cp.log(R.T @ b) + A.T @ lmbda) - c.T @ lmbda

        objective = cp.Maximize(wc_growth_rate)
        constraints = [
            cp.sum(b) == 1,
            b >= 0,
            lmbda >= 0,
        ]
        problem = cp.Problem(objective, constraints)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem.solve()
                return b.value
        except:
            return self.no_bet

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate optimal wager amounts in dollars
        """

        fractions = self.__calculate_optimal_wagers()
        wagers = self.fraction * self.current_bankroll * fractions[:-1]
        wagers_rounded = np.round(wagers, 2)
        wagers_clipped = np.where(wagers_rounded < self.min_bet, 0, wagers_rounded)

        red_wagers, blue_wagers = wagers_clipped[::2], wagers_clipped[1::2]

        return red_wagers, blue_wagers
