from dataclasses import dataclass
from enum import StrEnum

from .poke_engine import *


class Weather(StrEnum):
    NONE = "none"
    SUN = "sun"
    RAIN = "rain"
    SAND = "sand"
    HAIL = "hail"
    SNOW = "snow"
    HARSH_SUN = "harshsun"
    HEAVY_RAIN = "heavyrain"


class Terrain(StrEnum):
    NONE = "none"
    GRASSY = "grassyterrain"
    ELECTRIC = "electricterrain"
    MISTY = "mistyterrain"
    PSYCHIC = "psychicterrain"


class PokemonIndex(StrEnum):
    P0 = "0"
    P1 = "1"
    P2 = "2"
    P3 = "3"
    P4 = "4"
    P5 = "5"


@dataclass
class IterativeDeepeningResult:
    """
    Result of an Iterative Deepening Expectiminimax Search

    :param side_one: The moves for side_one
    :type side_one: list[str]
    :param side_two: The moves for side_two
    :type side_two: list[str]
    :param matrix: A vector representing the payoff matrix of the search.
        Pruned branches are represented by None
    :type matrix: int
    :param depth_searched: The depth that was searched to
    :type depth_searched: int
    """

    side_one: list[str]
    side_two: list[str]
    matrix: list[float]
    depth_searched: int

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=rust_result.s1,
            side_two=rust_result.s2,
            matrix=rust_result.matrix,
            depth_searched=rust_result.depth_searched,
        )

    def get_safest_move(self) -> str:
        """
        Get the safest move for side_one
        The safest move is the move that minimizes the loss for the turn

        :return: The safest move
        :rtype: str
        """
        safest_value = float("-inf")
        safest_s1_index = 0
        vec_index = 0
        for i in range(len(self.side_one)):
            worst_case_this_row = float("inf")
            for _ in range(len(self.side_two)):
                score = self.matrix[vec_index]
                if score < worst_case_this_row:
                    worst_case_this_row = score

            if worst_case_this_row > safest_value:
                safest_s1_index = i
                safest_value = worst_case_this_row

        return self.side_one[safest_s1_index]


@dataclass
class MctsSideResult:
    """
    Result of a Monte Carlo Tree Search for a single side

    :param move_choice: The move that was chosen
    :type move_choice: str
    :param total_score: The total score of the chosen move
    :type total_score: float
    :param visits: The number of times the move was chosen
    :type visits: int
    """

    move_choice: str
    total_score: float
    visits: int


@dataclass
class MctsResult:
    """
    Result of a Monte Carlo Tree Search

    :param side_one: Result for side one
    :type side_one: list[MctsSideResult]
    :param side_two: Result for side two
    :type side_two: list[MctsSideResult]
    :param total_visits: Total number of monte carlo iterations
    :type total_visits: int
    """

    side_one: list[MctsSideResult]
    side_two: list[MctsSideResult]
    total_visits: int

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=[
                MctsSideResult(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                )
                for i in rust_result.s1
            ],
            side_two=[
                MctsSideResult(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                )
                for i in rust_result.s2
            ],
            total_visits=rust_result.iteration_count,
        )


def monte_carlo_tree_search(state: State, duration_ms: int = 1000) -> MctsResult:
    """
    Perform monte-carlo-tree-search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search
    :type duration_ms: int
    :return: the result of the search
    :rtype: MctsResult
    """
    return MctsResult._from_rust(mcts(state, duration_ms))


def monte_carlo_tree_search_with_priors(
    state: State,
    s1_priors: list,
    s2_priors: list,
    duration_ms: int = 1000,
) -> MctsResult:
    """
    MCTS with policy net priors (PUCT selection).

    Priors bias the search toward moves the policy net thinks are good.
    Each prior list should match the number of available moves for that side.

    :param state: the state to search through
    :param s1_priors: prior probabilities for side_one's moves
    :param s2_priors: prior probabilities for side_two's moves
    :param duration_ms: time in milliseconds
    :return: the result of the search
    :rtype: MctsResult
    """
    return MctsResult._from_rust(mcts_with_priors(state, s1_priors, s2_priors, duration_ms))


def monte_carlo_tree_search_with_value(
    state: State,
    value_net,
    duration_ms: int = 1000,
    s1_priors: list = None,
    s2_priors: list = None,
    alpha: float = 1.0,
) -> MctsResult:
    """
    MCTS with value-net leaf evaluation (and optional PUCT priors).

    :param state: the state to search through
    :param value_net: a ValueNet instance (loaded ONNX model)
    :param duration_ms: time in milliseconds
    :param s1_priors: optional priors for side_one's moves
    :param s2_priors: optional priors for side_two's moves
    :param alpha: leaf-eval mixing weight (0=heuristic only, 1=value net only)
    :return: the result of the search
    :rtype: MctsResult
    """
    return MctsResult._from_rust(
        mcts_with_value(state, value_net, duration_ms, s1_priors, s2_priors, alpha)
    )


def monte_carlo_tree_search_multi(
    states: list, duration_ms: int = 1000
) -> MctsResult:
    """
    Perform MCTS over multiple sampled opponent states (root parallelization).

    Shares a single search tree but round-robins through the states each batch
    of iterations. This averages over uncertainty about the opponent's team
    without splitting the search budget.

    All states must have the same side_one and available moves.

    :param states: list of State objects (different opponent team samples)
    :type states: list[State]
    :param duration_ms: time in milliseconds to run the search
    :type duration_ms: int
    :return: the result of the search
    :rtype: MctsResult
    """
    return MctsResult._from_rust(mcts_multi(states, duration_ms))


def iterative_deepening_expectiminimax(
    state: State, duration_ms: int = 1000
) -> IterativeDeepeningResult:
    """
    Perform an iterative-deepening expectiminimax search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search
    :type duration_ms: int
    :return: the result of the search
    :rtype: IterativeDeepeningResult
    """
    return IterativeDeepeningResult._from_rust(id(state, duration_ms))
