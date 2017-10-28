"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent

from importlib import reload


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
        self.player1 = game_agent.MinimaxPlayer()
        self.player2 = game_agent.MinimaxPlayer()
        self.game = isolation.Board(self.player1, self.player2)

    def test_minimax_decision(self):
        best_move = self.player1.minimax(self.game, 3)

        self.assertIsNotNone(best_move)


class ScoreByNrPossibleMovesTests(unittest.TestCase):
    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.MinimaxPlayer()
        self.player2 = game_agent.MinimaxPlayer()
        self.game = isolation.Board(self.player1, self.player2)

    def test_score_by_nr_possible_moves(self):
        score = lambda p: game_agent.score_by_nr_possible_moves(self.game, p)
        self.assertEqual(score(self.player1), 49)
        self.assertEqual(score(self.player2), 49)
        # [[2, 1], [2, 5], [1, 3], [4, 6]]
        # Player 1 moves
        self.game.apply_move((2, 1))
        self.assertEqual(score(self.player1), 6)
        self.assertEqual(score(self.player2), 48)
        # Player 2 moves
        self.game.apply_move((2, 5))
        self.assertEqual(score(self.player1), 6)
        self.assertEqual(score(self.player2), 6)


if __name__ == '__main__':
    unittest.main()
