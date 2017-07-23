"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
# import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Maximize open moves while staying away from center. If availability drops,
    maximize move difference between players.

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
    # Set number of available moves
    own_moves = len(game.get_legal_moves(player))
    # set player and center locations
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    # set distance to center
    center = float((h - y) ** 2 + (w - x) ** 2)
    if center == 0:
        return 0
    else:
        return float((own_moves*1.5) / center)


def custom_score_2(game, player):
    """Maximize open move difference while staying away from the center of the board.

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
    # set player available moves
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    # set player and center locations
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    # set distance to center
    center = float((h - y) ** 2 + (w - x) ** 2)
    if center != 0:
        return float((own_moves-opp_moves) / center)
    else:
        return float(own_moves-opp_moves)


def custom_score_3(game, player):
    """Maximize open moves while gravitating toward the center of the board,
    so long as the move availability is high. If availability drops, maximize
    move difference between players.

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
    # set player available moves
    own_moves = len(game.get_legal_moves(player))
    # set player and center locations
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    # set distance to center
    center = float((h - y) ** 2 + (w - x) ** 2)
    if center != 0:
        # this will prevent it from getting stuck on center
        return float(own_moves * (center/1.5))
    else:
        return float(own_moves)


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

    def max_value(self, game, depth):
        """
        Find the max value for next depth's nodes. Max represents the IsolationPlayer's turn.
        :param game: isolation.Board instance
        :param depth: current depth, either from previous min_val call, or from original minimax call
        :return: score for terminal leaf node
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Terminal check
        moves = game.get_legal_moves()
        if not moves or depth == 0:
            return self.score(game, self)
        else:
            v = float('-inf')
            # find max value of next depth's nodes
            for action in moves:
                v = max(v, self.min_value(game.forecast_move(action), depth-1))
            return v

    def min_value(self, game, depth):
        """
        Find the min value for next depth's nodes. Min represents the Opponent's turn.
        :param game: isolation.Board instance
        :param depth: current depth, either from previous max_val call, or from original minimax call
        :return: score for terminal leaf node
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # terminal check
        moves = game.get_legal_moves()
        if not moves or depth == 0:
            return self.score(game, self)
        else:
            # find min value of next depth's nodes
            v = float('inf')
            for action in moves:
                v = min(v, self.max_value(game.forecast_move(action), depth-1))
        return v

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
            (1) You MUST use the `self.score(game, self)` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        if len(moves) > 1:
            if depth != 0:
                min_results = {}
                # start down the tree to get the action in the next depth that has the max score
                for action in moves:
                    min_results[action] = self.min_value(game.forecast_move(action), depth-1)
                return max(min_results, key=min_results.get)
            else:
                # depth 0, so just get the action with the max score
                max_results = {}
                for action in moves:
                    max_results[action] = self.score(game.forecast_move(action), game.active_player)
                return max(max_results, key=max_results.get)
        else:
            if len(moves) == 1:
                return moves[0]
            else:
                return -1, -1


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    def alpha_max(self, game, depth, alpha, beta):
        """
        Find the max value for next depth's nodes. Max represents the IsolationPlayer's turn.
        :param game: isolation.Board instance
        :param depth: current depth, either from previous min_val call, or from original minimax call
        :param alpha: current maximum value, used for pruning
        :param beta: current minimum value, used for pruning
        :return: score for terminal leaf node
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Terminal check
        if not game.get_legal_moves() or depth == 0:
            return self.score(game, self)
        else:
            v = float('-inf')
            # find max value of next depth's nodes
            for action in game.get_legal_moves():
                v = max(v, self.alpha_min(game.forecast_move(action), depth-1, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

    def alpha_min(self, game, depth, alpha, beta):
        """
        Find the min value for next depth's nodes. Min represents the Opponent's turn.
        :param game: isolation.Board instance
        :param depth: current depth, either from previous max_val call, or from original minimax call
        :param alpha: current maximum value, used for pruning
        :param beta: current minimum value, used for pruning
        :return: score for terminal leaf node
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # terminal check
        if not game.get_legal_moves() or depth == 0:
            return self.score(game, self)
        else:
            # find min value of next depth's nodes
            v = float('inf')
            for action in game.get_legal_moves():
                v = min(v, self.alpha_max(game.forecast_move(action), depth-1, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
        return v

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
        # Initialize the best move to the best score in depth 0 so that this function returns something
        # in case the search fails due to timeout
        moves = game.get_legal_moves()
        moves_length = len(moves)
        if moves_length > 0:
            # best_current = moves[random.randint(0, len(moves) - 1)]
            max_results = {}
            for action in moves:
                max_results[action] = self.score(game.forecast_move(action), game.active_player)
            best_current = max(max_results, key=max_results.get)
            best_move = best_current
        elif moves_length == 1:
            return moves[0]
        else:
            return -1, -1
        # The try/except block will automatically catch the exception
        # raised when the timer is about to expire.
        try:
            # set depth depending on open moves.
            if moves_length >= 3:
                depth = 1
            else:
                depth = 2
            while True:
                best_move = self.alphabeta(game, depth)
                if best_move == (-1, -1):
                    # AB returned a losing end-state from a deeper tree, return best of depth 0 instead
                    best_move = best_current
                depth += 1
        except SearchTimeout:
            # Handle any actions required after timeout as needed
            pass
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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        try:
            # initialize the search tree
            v = self.alpha_max(game, depth, alpha, beta)
            # start down the tree to get the action in the next depth that has the max score
            for action in moves:
                iteration = self.alpha_min(game.forecast_move(action), depth - 1, alpha, beta)
                if iteration == v:
                    return action
                alpha = max(alpha, iteration)
        except:
            # If iteration never = v, timeout, which will default to best score in depth 0
            raise SearchTimeout()
