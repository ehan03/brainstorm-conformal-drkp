# standard library imports
import os
from typing import Union

# third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from joblib import load

# local imports
from kelly import DistributionalRobustKelly, NaiveKelly
from venn_abers import VennAbersCV


class BacktestFramework:
    """
    Class for evaluating betting performance of the models over events
    from 2022 to early 2024
    """

    DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
    MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")

    def __init__(self, initial_bankroll: float = 100.0) -> None:
        """
        Initialize the BacktestFramework class
        """

        self.initial_bankroll = initial_bankroll
        self.test_df = pd.read_csv(os.path.join(self.DATA_PATH, "test.csv")).drop(
            columns=["DATE", "BOUT_ORDINAL", "RED_WIN"]
        )
        self.odds_df = pd.read_csv(os.path.join(self.DATA_PATH, "backtest_odds.csv"))

        # Load prefit models
        self.lr = load(os.path.join(self.MODELS_PATH, "logreg.joblib"))
        self.va_lr = load(os.path.join(self.MODELS_PATH, "venn_abers_logreg.joblib"))

    def helper_calculate_profit(
        self, wager_side: str, red_win: Union[int, float], wager: float, odds: float
    ) -> float:
        if np.isnan(red_win) or wager == 0.0:
            return 0.0

        if wager_side == "RED":
            return wager * red_win * odds - wager
        else:
            return wager * (1 - red_win) * odds - wager

    def run_backtest(self):
        event_ids = self.odds_df["EVENT_ID"].unique()

        results_dict = {
            "DATES": [pd.to_datetime("2022-01-01")],
            "BANKROLL_LR": [self.initial_bankroll],
            "BANKROLL_VA+LR": [self.initial_bankroll],
            "BANKROLL_DUMMY": [self.initial_bankroll],
            "TOTAL_WAGER_LR": [0.0],
            "TOTAL_WAGER_VA+LR": [0.0],
            "TOTAL_WAGER_DUMMY": [0.0],
            "ROI_LR": [0.0],
            "ROI_VA+LR": [0.0],
            "ROI_DUMMY": [0.0],
        }

        for event_id in event_ids:
            X_test = self.test_df.loc[self.test_df["EVENT_ID"] == event_id].copy()
            X_test = X_test.drop(columns=["EVENT_ID", "BOUT_ID"])
            sliced_df = self.odds_df.loc[self.odds_df["EVENT_ID"] == event_id].copy()
            red_odds = sliced_df["RED_FIGHTER_ODDS"].to_numpy()
            blue_odds = sliced_df["BLUE_FIGHTER_ODDS"].to_numpy()
            results_dict["DATES"].append(pd.to_datetime(sliced_df["DATE"].values[0]))

            for model in ["LR", "VA+LR", "DUMMY"]:
                current_bankroll = results_dict[f"BANKROLL_{model}"][-1]

                if model == "DUMMY":
                    red_wagers, blue_wagers = [], []

                    for red_odd, blue_odd in zip(red_odds, blue_odds):
                        wager = np.round(current_bankroll * 0.01, 2)
                        if red_odd < blue_odd:
                            red_wagers.append(wager)
                            blue_wagers.append(0)
                        else:
                            red_wagers.append(0)
                            blue_wagers.append(wager)

                    red_wagers, blue_wagers = np.array(red_wagers), np.array(
                        blue_wagers
                    )
                elif model == "LR":
                    probs = self.lr.predict_proba(X_test)
                    red_probs = probs[:, 1]
                    blue_probs = probs[:, 0]

                    naive_kelly = NaiveKelly(
                        red_probs=red_probs,
                        blue_probs=blue_probs,
                        red_odds=red_odds,
                        blue_odds=blue_odds,
                        current_bankroll=current_bankroll,
                    )

                    red_wagers, blue_wagers = naive_kelly()
                else:
                    _, p0_p1 = self.va_lr.predict_proba(X_test.to_numpy())

                    robust_kelly = DistributionalRobustKelly(
                        p0_p1=p0_p1,
                        red_odds=red_odds,
                        blue_odds=blue_odds,
                        current_bankroll=current_bankroll,
                    )

                    red_wagers, blue_wagers = robust_kelly()

                sliced_df[f"RED_WAGER_{model}"] = red_wagers
                sliced_df[f"BLUE_WAGER_{model}"] = blue_wagers

                total_wager = (
                    sliced_df[f"RED_WAGER_{model}"].sum()
                    + sliced_df[f"BLUE_WAGER_{model}"].sum()
                )
                results_dict[f"TOTAL_WAGER_{model}"].append(
                    round(results_dict[f"TOTAL_WAGER_{model}"][-1] + total_wager, 2)
                )

                sliced_df[f"RED_PROFIT_{model}"] = sliced_df.apply(
                    lambda row: self.helper_calculate_profit(
                        "RED",
                        row["RED_WIN"],
                        row[f"RED_WAGER_{model}"],
                        row["RED_FIGHTER_ODDS"],
                    ),
                    axis=1,
                ).round(2)
                sliced_df[f"BLUE_PROFIT_{model}"] = sliced_df.apply(
                    lambda row: self.helper_calculate_profit(
                        "BLUE",
                        row["RED_WIN"],
                        row[f"BLUE_WAGER_{model}"],
                        row["BLUE_FIGHTER_ODDS"],
                    ),
                    axis=1,
                ).round(2)

                total_profit = (
                    sliced_df[f"RED_PROFIT_{model}"].sum()
                    + sliced_df[f"BLUE_PROFIT_{model}"].sum()
                )
                results_dict[f"BANKROLL_{model}"].append(
                    round(results_dict[f"BANKROLL_{model}"][-1] + total_profit, 2)
                )

                roi = (
                    100.0
                    * (results_dict[f"BANKROLL_{model}"][-1] - self.initial_bankroll)
                    / results_dict[f"TOTAL_WAGER_{model}"][-1]
                    if results_dict[f"TOTAL_WAGER_{model}"][-1] > 0
                    else 0.0
                )
                results_dict[f"ROI_{model}"].append(roi)

        return pd.DataFrame(results_dict)

    def plot_bankrolls_over_time(self, results_df: pd.DataFrame) -> None:
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.step(
            results_df["DATES"],
            results_df["BANKROLL_LR"],
            label="Logistic Regression",
            color="#ff8389",
        )
        ax.step(
            results_df["DATES"],
            results_df["BANKROLL_VA+LR"],
            label="Venn Abers + Logistic Regression",
            color="#1192e8",
        )
        ax.step(
            results_df["DATES"],
            results_df["BANKROLL_DUMMY"],
            label="Dummy",
            color="#008000",
        )
        ax.hlines(
            y=self.initial_bankroll,
            xmin=results_df["DATES"].min(),
            xmax=results_df["DATES"].max(),
            color="grey",
            linestyle="--",
        )
        ax.hlines(
            y=0,
            xmin=results_df["DATES"].min(),
            xmax=results_df["DATES"].max(),
            color="red",
            linestyle="--",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Bankroll ($)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_roi_over_time(self, results_df: pd.DataFrame) -> None:
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.step(
            results_df["DATES"],
            results_df["ROI_LR"],
            label="Logistic Regression",
            color="#ff8389",
        )
        ax.step(
            results_df["DATES"],
            results_df["ROI_VA+LR"],
            label="Venn Abers + Logistic Regression",
            color="#1192e8",
        )
        ax.step(
            results_df["DATES"],
            results_df["ROI_DUMMY"],
            label="Dummy",
            color="#008000",
        )
        ax.hlines(
            y=0,
            xmin=results_df["DATES"].min(),
            xmax=results_df["DATES"].max(),
            color="grey",
            linestyle="--",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("ROI (%)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def __call__(self) -> None:
        results_df = self.run_backtest()
        self.plot_bankrolls_over_time(results_df)
        self.plot_roi_over_time(results_df)

        # Display the final results
        display(
            results_df[
                [
                    "BANKROLL_LR",
                    "BANKROLL_VA+LR",
                    "BANKROLL_DUMMY",
                    "ROI_LR",
                    "ROI_VA+LR",
                    "ROI_DUMMY",
                ]
            ].tail(1)
        )
