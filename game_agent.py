"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

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
    
    # This heuristic function has a tie-break feature, which
    # gives a bit more weight to moves that take the player to a state where
    # the number of blank spaces is an odd number, which from experience from
    # playing the game, I believe it's advantageous in most cases
    
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    
    if len(game.get_blank_spaces()) % 2 == 0:
        bonus = 0
    else:
        bonus = 0.5      
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves + bonus)

def custom_score1(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2*opp_moves)

def custom_score2(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    
    if len(game.get_blank_spaces()) % 2 == 0:
        bonus = 0
    else:
        bonus = 0.5  
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves + bonus)

def custom_score3(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    
    if len(game.get_blank_spaces()) % 2 == 0:
        bonus = 0
    else:
        bonus = 0.5 
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2*opp_moves + bonus)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        
    # This will be our method to assess the time left, and raise the
    # timeout exception if time is close to running out
    def assess_time_left(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function performs iterative deepening if self.iterative=True,
        and it uses the search method (minimax or alphabeta) corresponding
        to the self.method value.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

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
                
        self.assess_time_left()
        
        # If there are no legal moves, return (-1, -1)
        if not legal_moves:
            return (-1, -1)
        
        # Do a depth limited search using a given depth level, choose the
        # appropriate method corresponding to self.method value.
        # This method saves the last best_move computed.
        def depth_limited_search(self, game, depth):
            self.assess_time_left()
            if self.method == 'minimax':
                best_score, best_move = self.minimax(game, depth)
            elif self.method == 'alphabeta':
                best_score, best_move = self.alphabeta(game, depth)
            self.best_move = best_move
            return best_score, best_move
        
        # Do an iterative deepening search, iterating over the 
        # last function: depth_limited_search, passing it depths starting
        # from 1 up to infinity. This method will come to a halt when 
        # the timeout exception is called or when the game enters a 
        # final game state.
        def iterative_deepening_search(self, game):
            best_move = (-1,-1)
            depth = 1;
            while True:
                best_score, best_move = depth_limited_search(self, game, depth)
                depth += 1
                if best_score == float("inf") or best_score == float("-inf"):
                    return best_move
            return best_move
        
        # Run an iterative deepening search if self.iterative is true,
        # or a depth limited search otherwise.
        try:
            if self.iterative:
                result = iterative_deepening_search(self, game)
            elif not self.iterative:
                best_score, result = depth_limited_search(self, game, self.search_depth)
        
        # When the timeout exception is raised, return the last best_move found
        except Timeout:
            return self.best_move
        
        # Return the best move from the last completed search iteration
        return result
        
        
    def minimax(self, game, depth, maximizing_player=True):
        """Calculate the best move by searching until given depth using 
        a minimax method

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        self.assess_time_left()
              
        def max_value(game, depth):
            self.assess_time_left()
            # Check if depth limit has been reached or if game 
            # is in terminal state, if so, return utility value
            if depth==1 or game.is_loser(self) or game.is_winner(self):
                return self.score(game, self)
            moves = game.get_legal_moves()
            best_score = float("-inf")
            for move in moves:
                child = game.forecast_move(move)
                # assign best_score between max of previous value or 
                # the value thrown by the minimax-min_value function
                best_score = max(best_score, min_value(child, depth-1))
            return best_score
        
        def min_value(game, depth):
            self.assess_time_left()
            if depth==1 or game.is_loser(self) or game.is_winner(self):
                return self.score(game, self)
            moves = game.get_legal_moves()
            best_score = float("inf")
            for move in moves:
                child = game.forecast_move(move)
                # assign best_score between min of previous value or 
                # the value thrown by the minimax-max_value function
                best_score = min(best_score, max_value(child, depth-1))
            return best_score
        
        start_moves = game.get_legal_moves()
        # get a list of scores from each child node using minimax.
        scores = [min_value(game.forecast_move(m), depth) for m in start_moves]
        # return the value of the best score, and the move associated to it.
        return max(scores), start_moves[scores.index(max(scores))]
    
        

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Calculate the best move by searching until given depth using 
        an alphabeta method.

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

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        self.assess_time_left()

        def max_value(game, depth, alpha, beta):
            self.assess_time_left()
            # Check if depth limit has been reached or if game 
            # is in terminal state, if so, return utility value and last move
            if depth==0 or game.is_loser(self) or game.is_winner(self):
                return self.score(game, self), game.get_player_location(game.active_player)
            moves = game.get_legal_moves()
            best_score = float("-inf")
            best_move = (-1,-1)
            for move in moves:
                child = game.forecast_move(move)
                # get the best score for current branch, 
                # the move returned will not be used
                score, m = min_value(child, depth-1, alpha, beta)
                # check if this score is better than previous best score
                # if so, update best score and best move
                if score > best_score:
                    best_score = score
                    best_move = move
                # as beta is the upper bound of search, if best_score 
                # is greater than beta, return best_score and best_move without
                # checking other nodes (child) under this node (move)
                if best_score >= beta:
                    return best_score, best_move
                # if best_score is lesser than beta, 
                # update the lower bound of search (alpha) and continue with 
                # the search at the current depth
                alpha = max(alpha, best_score)
            return best_score, best_move
        
        def min_value(game, depth, alpha, beta):
            self.assess_time_left()
            if depth==0 or game.is_loser(self) or game.is_winner(self):
                return self.score(game, self), game.get_player_location(game.active_player)
            moves = game.get_legal_moves()
            best_score = float("inf")
            best_move = (-1,-1)
            for move in moves:
                child = game.forecast_move(move)
                score, m = max_value(child, depth-1, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = move
                if best_score <= alpha:
                    return best_score, best_move
                beta = min(beta, best_score)
            return best_score, best_move
        
        # return best_score and its associated best_move
        return max_value(game, depth, alpha, beta)
        
