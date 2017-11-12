"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return Value2WeightedScore(game, player)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return Value2Score(game, player)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return ValueWeightedScore(game, player)


def opening_score(location):
    if location == (4, 4):
        return float('inf')
    else:
        return float('-inf')


CELL_VALUES = {(0, 0): 0.25, (0, 1): 0.38, (0, 2): 0.5, (0, 3): 0.5, (0, 4): 0.5, (0, 5): 0.38, (0, 6): 0.25,
               (1, 0): 0.38, (1, 1): 0.5, (1, 2): 0.75, (1, 3): 0.75, (1, 4): 0.75, (1, 5): 0.5, (1, 6): 0.38,
               (2, 0): 0.5, (2, 1): 0.75, (2, 2): 1.0, (2, 3): 1.0, (2, 4): 1.0, (2, 5): 0.75, (2, 6): 0.5, (3, 0): 0.5,
               (3, 1): 0.75, (3, 2): 1.0, (3, 3): 1.0, (3, 4): 1.0, (3, 5): 0.75, (3, 6): 0.5, (4, 0): 0.5,
               (4, 1): 0.75, (4, 2): 1.0, (4, 3): 1.0, (4, 4): 1.0, (4, 5): 0.75, (4, 6): 0.5, (5, 0): 0.38,
               (5, 1): 0.5, (5, 2): 0.75, (5, 3): 0.75, (5, 4): 0.75, (5, 5): 0.5, (5, 6): 0.38, (6, 0): 0.25,
               (6, 1): 0.38, (6, 2): 0.5, (6, 3): 0.5, (6, 4): 0.5, (6, 5): 0.38, (6, 6): 0.25}

CELL_VALUES_SQ = {(0, 0): 0.07, (0, 1): 0.15, (0, 2): 0.25, (0, 3): 0.25, (0, 4): 0.25, (0, 5): 0.15, (0, 6): 0.07,
                  (1, 0): 0.15, (1, 1): 0.25, (1, 2): 0.57, (1, 3): 0.57, (1, 4): 0.57, (1, 5): 0.25, (1, 6): 0.15,
                  (2, 0): 0.25, (2, 1): 0.57, (2, 2): 1.0, (2, 3): 1.0, (2, 4): 1.0, (2, 5): 0.57, (2, 6): 0.25,
                  (3, 0): 0.25, (3, 1): 0.57, (3, 2): 1.0, (3, 3): 1.0, (3, 4): 1.0, (3, 5): 0.57, (3, 6): 0.25,
                  (4, 0): 0.25, (4, 1): 0.57, (4, 2): 1.0, (4, 3): 1.0, (4, 4): 1.0, (4, 5): 0.57, (4, 6): 0.25,
                  (5, 0): 0.15, (5, 1): 0.25, (5, 2): 0.57, (5, 3): 0.57, (5, 4): 0.57, (5, 5): 0.25, (5, 6): 0.15,
                  (6, 0): 0.07, (6, 1): 0.15, (6, 2): 0.25, (6, 3): 0.25, (6, 4): 0.25, (6, 5): 0.15, (6, 6): 0.07}

CELL_VALUES_POW3 = {(0, 0): 0.02, (0, 1): 0.06, (0, 2): 0.13, (0, 3): 0.13, (0, 4): 0.13, (0, 5): 0.06, (0, 6): 0.02,
                    (1, 0): 0.06, (1, 1): 0.13, (1, 2): 0.43, (1, 3): 0.43, (1, 4): 0.43, (1, 5): 0.13, (1, 6): 0.06,
                    (2, 0): 0.13, (2, 1): 0.43, (2, 2): 1.0, (2, 3): 1.0, (2, 4): 1.0, (2, 5): 0.43, (2, 6): 0.13,
                    (3, 0): 0.13, (3, 1): 0.43, (3, 2): 1.0, (3, 3): 1.0, (3, 4): 1.0, (3, 5): 0.43, (3, 6): 0.13,
                    (4, 0): 0.13, (4, 1): 0.43, (4, 2): 1.0, (4, 3): 1.0, (4, 4): 1.0, (4, 5): 0.43, (4, 6): 0.13,
                    (5, 0): 0.06, (5, 1): 0.13, (5, 2): 0.43, (5, 3): 0.43, (5, 4): 0.43, (5, 5): 0.13, (5, 6): 0.06,
                    (6, 0): 0.02, (6, 1): 0.06, (6, 2): 0.13, (6, 3): 0.13, (6, 4): 0.13, (6, 5): 0.06, (6, 6): 0.02}


def SumCellValuesPlayer(game, player):
    player_moves_value = moves_value(game, player, CELL_VALUES)
    return player_moves_value

def ValueScore(game, player):
    return moves_value(game, player, CELL_VALUES) - moves_value(game, game.get_opponent(player), CELL_VALUES)


def ValueWeightedScore(game, player):
    return moves_value(game, player, CELL_VALUES) - 2 * moves_value(game, game.get_opponent(player), CELL_VALUES)


def SumCellValuesSqPlayer(game, player):
    player_moves_value = moves_value(game, player, CELL_VALUES_SQ)
    return player_moves_value


def Value2Score(game, player):
    return moves_value(game, player, CELL_VALUES_SQ) - moves_value(game, game.get_opponent(player),
                                                                   CELL_VALUES_SQ)


def Value2WeightedScore(game, player):
    return moves_value(game, player, CELL_VALUES_SQ) - 2 * moves_value(game, game.get_opponent(player),
                                                                       CELL_VALUES_SQ)


def Value3Score(game, player):
    return moves_value(game, player, CELL_VALUES_POW3) - moves_value(game, game.get_opponent(player),
                                                                     CELL_VALUES_POW3)


def Value3WeightedScore(game, player):
    return moves_value(game, player, CELL_VALUES_POW3) - 2 * moves_value(game, game.get_opponent(player),
                                                                         CELL_VALUES_POW3)


def moves_value(game, player, values):
    player_moves = game.get_legal_moves(player)
    return float(sum(map(lambda m: values[m], player_moves)))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        super().__init__(search_depth, score_fn, timeout)
        self.time_left = lambda: float('inf')

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # assert game.active_player == self, "unexpected game state"
        return self.max_value(game, depth)[0];

    def min_value(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        return self.value(game, depth, self.max_value, min)

    def max_value(self, game, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        return self.value(game, depth, self.min_value, max)

    def value(self, game, depth, child_value, optimal_value):
        """
        Evaluates the value a game state
        :param game: the game which state is being evaluated
        :param depth: the depth in the game-tree
        :param child_value: a function to retrieve the child game states in the game tree
        :param optimal_value: a function to choose the optimal value of all child game states
        :return: the value of the game state
        """
        legal_moves = game.get_legal_moves()
        if self.is_terminal(game, legal_moves, depth):
            return (None, self.score(game, self))

        values = map(lambda m: (m, child_value(game.forecast_move(m), depth - 1)[1]), legal_moves)
        value = optimal_value(values, key=lambda t: t[1])
        return value

    def is_terminal(self, game, legal_moves, depth):
        """ Checks if the at this level of the game tree a score should be returned.
        :return : True for a score should be returned, False for continuing evaluating down the game tree.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # either depth reached for estimation or game is over
        return depth == 0 or not any(legal_moves)


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        super().__init__(search_depth, score_fn, timeout)
        self.time_left = lambda: float('inf')

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            depth = 1
            # Iterative deepening
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        return self.max_value(game, depth, alpha, beta)[0]

    def min_value(self, game, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        # return self.value(game, depth, alpha, beta, self.max_value, lambda u, b: u < b)
        legal_moves = game.get_legal_moves()
        if self.is_terminal(game, legal_moves, depth):
            return (None, self.score(game, self))

        optimal_move = None
        for move in legal_moves:
            value = self.max_value(game.forecast_move(move), depth - 1, alpha, beta)[1]
            if optimal_move == None or value < optimal_move[1]:
                optimal_move = (move, value)
            if value <= alpha:
                return optimal_move
            beta = min(beta, value)

        return optimal_move

    def max_value(self, game, depth, alpha, beta):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        # return self.value(game, depth, alpha, beta, self.min_value, lambda u, b: u > b)
        legal_moves = game.get_legal_moves()
        if self.is_terminal(game, legal_moves, depth):
            return (None, self.score(game, self))

        optimal_move = None
        for move in legal_moves:
            value = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)[1]
            if optimal_move == None or value > optimal_move[1]:
                optimal_move = (move, value)
            if value >= beta:
                return optimal_move
            alpha = max(alpha, value)

        return optimal_move

    def is_terminal(self, game, legal_moves, depth):
        """ Checks if the at this level of the game tree a score should be returned.
        :return : True for a score should be returned, False for continuing evaluating down the game tree.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # either depth reached for estimation or game is over
        return depth == 0 or not any(legal_moves)
