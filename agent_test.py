"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import sample_players
import math

from importlib import reload


class ExpansionRecorder:
    ''' Records how an algorithm expands the game-tree'''

    def __init__(self):
        self._moves = set()

    def add(self, move):
        self._moves.add(move)

    @property
    def moves(self):
        return self._moves


class BoardSpy(isolation.Board):
    ''' A Spy in order that we can observer and test algorithms '''

    def __init__(self, player_1, player_2, expansion_recorder, width=7, height=7):
        super().__init__(player_1, player_2, width, height)
        self.expansion_recorder = expansion_recorder

    def create_board(self):
        return BoardSpy(self._player_1, self._player_2, self.expansion_recorder, width=self.width, height=self.height)

    def forecast_move(self, move):
        self.expansion_recorder.add(move)
        return super().forecast_move(move)


def create_board_with_state(player1, player2, board_state, expansion_recorder=ExpansionRecorder()):
    length = math.sqrt(len(board_state) - 3)
    game = BoardSpy(player1, player2, expansion_recorder, width=int(length), height=int(length))
    game._board_state = board_state
    return game


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)


class MinimaxPlayerTests(unittest.TestCase):
    def setUp(self):
        reload(game_agent)

    def test_minimax_decision(self):
        player_factory = lambda: game_agent.MinimaxPlayer(search_depth=1, score_fn=sample_players.open_move_score)
        player1 = player_factory()
        player2 = player_factory()
        game = isolation.Board(player1, player2)
        best_move = player1.minimax(game, 3)

        self.assertIsNotNone(best_move)

    def test_minimax_WhenDepth1(self):
        player_factory = lambda: game_agent.MinimaxPlayer(search_depth=1, score_fn=sample_players.open_move_score)
        player1 = player_factory()
        player2 = player_factory()
        board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
                       1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 20]
        game = create_board_with_state(player1, player2, board_state)

        best_move = player1.minimax(game, player1.search_depth)

        self.assertIn(best_move, [(0, 3)])

    def test_minimax_WhenDepth3(self):
        player_factory = lambda: game_agent.MinimaxPlayer(search_depth=3, score_fn=sample_players.open_move_score)
        player1 = player_factory()
        player2 = player_factory()
        board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                       1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 20]
        game = create_board_with_state(player1, player2, board_state)

        best_move = player1.minimax(game, player1.search_depth)

        self.assertIn(best_move, [(3, 0)])

    def test_minimax_WhenIsAWholeGame(self):
        player_factory = lambda: game_agent.MinimaxPlayer(search_depth=3, score_fn=sample_players.open_move_score)
        player1 = player_factory()
        player2 = player_factory()
        board = isolation.Board(player1, player2)
        board.play(10000)


class AlphaBetaPlayerTests(unittest.TestCase):
    def setUp(self):
        reload(game_agent)

    def test_alphabeta_WhenDepth1(self):
        player_factory = lambda: game_agent.AlphaBetaPlayer(search_depth=1, score_fn=sample_players.open_move_score)
        player1 = player_factory()
        player2 = player_factory()
        board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
                       1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 20]
        game = create_board_with_state(player1, player2, board_state)

        best_move = player1.alphabeta(game, player1.search_depth)

        self.assertIn(best_move, [(0, 3)])

    def test_alphabeta_WhenDepth3(self):
        player_factory = lambda: game_agent.AlphaBetaPlayer(search_depth=3, score_fn=sample_players.open_move_score)
        player1 = player_factory()
        player2 = player_factory()
        board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                       1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 20]
        game = create_board_with_state(player1, player2, board_state)

        best_move = player1.alphabeta(game, player1.search_depth)

        self.assertIn(best_move, [(3, 0)])

    def test_alphabeta(self):
        expansion_recorder = ExpansionRecorder()
        player_factory = lambda: game_agent.AlphaBetaPlayer(search_depth=2, score_fn=sample_players.open_move_score)
        player1 = player_factory()
        player2 = player_factory()
        board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                       1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 23]
        game = create_board_with_state(player1, player2, board_state, expansion_recorder)

        player1.alphabeta(game, player1.search_depth)

        self.assertSetEqual(expansion_recorder.moves,
                            {(6, 4), (7, 3), (2, 6), (3, 3), (7, 1), (4, 4), (5, 7), (6, 0), (3, 7), (4, 0)},
                            "unexpected nodes were expanded")

    def test_get_move_WhenNoTimeRestriction(self):
        expansion_recorder = ExpansionRecorder()
        player_factory = lambda: game_agent.AlphaBetaPlayer(search_depth=2, score_fn=sample_players.open_move_score)
        player1 = player_factory()
        player2 = player_factory()
        board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                       1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 23]
        game = create_board_with_state(player1, player2, board_state, expansion_recorder)

        time_left = float('inf')
        move = player1.get_move(game, lambda: time_left)

        self.assertEqual(move, (3, 3))


if __name__ == '__main__':
    unittest.main()
