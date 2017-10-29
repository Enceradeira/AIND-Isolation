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


def create_board_with_state(player1, player2, board_state):
    length = math.sqrt(len(board_state) - 3)
    game = isolation.Board(player1, player2, width=int(length), height=int(length))
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
        player1.Name = 'Max'
        player2 = player_factory()
        player2.Name = 'Min'
        board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
                       1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 20]
        game = create_board_with_state(player1, player2, board_state)

        best_move = player1.minimax(game, player1.search_depth)

        self.assertIn(best_move, [(0, 3)])

    def test_minimax_WhenDepth3(self):
        player_factory = lambda: game_agent.MinimaxPlayer(search_depth=3, score_fn=sample_players.open_move_score)
        player1 = player_factory()
        player1.Name = 'Max'
        player2 = player_factory()
        player2.Name = 'Min'
        board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                       1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 20]
        game = create_board_with_state(player1, player2, board_state)

        print(game.to_string())

        best_move = player1.minimax(game, player1.search_depth)

        self.assertIn(best_move, [(3, 0)])


if __name__ == '__main__':
    unittest.main()
