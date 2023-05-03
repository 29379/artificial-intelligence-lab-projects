from reversi import *
from errors import *
from board import *
from copy import deepcopy
import numpy as np

class Algorithms:
    def __init__(self, depth: int, heuristic: any):
        self.heuristic = heuristic
        self.depth = depth
        
# def minimax(self, board, curDepth, prev_move, maxTurn, targetDepth, player):
# if curDepth == targetDepth or board.game_state != 'In progress' \
#         or (len(board.current_player_available_moves()) == 0 and prev_move is not None):

#     if board.game_state == player:
#         return float('inf'), prev_move

#     elif board.game_state == 'Tie':
#         return 0, prev_move

#     elif board.game_state != 'In progress':
#         return -float('inf'), prev_move

#     return self.heuristic(board, player), prev_move
        
        
    def minmax(self, reversi: Reversi, depth: int, max_depth: int, prev_move: any, max_turn: bool):
        if depth == max_depth or \
            reversi.check_game_status() != reversi.GAME_STATUS.get('IN_PROGRESS') or \
            len(reversi.board.get_valid_moves(reversi.board.current_player)) == 0 and prev_move is not None:
                if reversi.check_game_status() == reversi.GAME_STATUS.get('Tie'):
                    return 0, prev_move
                elif reversi.check_game_status() == reversi.GAME_STATUS.get('WHITE_WIN'):
                        if reversi.board.current_player == reversi.board.WHITE:
                            return float('inf'), prev_move
                        else:
                            return -float('inf'), prev_move
                elif reversi.check_game_status() == reversi.GAME_STATUS.get('WHITE_WIN'):
                        if reversi.board.current_player == reversi.board.BLACK:
                            return float('inf'), prev_move
                        else:
                            return -float('inf'), prev_move 
                else:
                    return self.heuristic(reversi), prev_move
                
        if max_turn:
            pass   
                
            
        
        
            
            
    
    
    def alpha_beta_pruning():
        pass
    