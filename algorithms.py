from reversi import *
from errors import *
from board import *
from copy import deepcopy
import numpy as np

class Algorithms:
    def __init__(self, depth: int, heuristic: any):
        self.heuristic = heuristic
        self.max_depth = depth
            
        
    def minmax(self, reversi: Reversi, depth: int, prev_move: tuple[int, int], max_turn: bool) -> tuple[float, tuple[int, int]]:
        if depth == self.max_depth or \
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
            best_score = -float('inf')
            best_move = None
            valid_moves = reversi.board.get_valid_moves(reversi.board.current_player)
             
            for move in valid_moves:
                new_board = deepcopy(reversi)
                prev_move = move if depth == 0 else prev_move
                new_board.play(move)
                #   go into recusion
                score, candidate = self.minmax(
                    new_board, 
                    depth + 1,
                    prev_move, 
                    False 
                )
                
                #   why???
                if score == float('inf'):
                    return score, candidate
                elif score > best_score:
                    best_score = score
                    best_move = candidate
            
            if best_move is None:
                return best_score, prev_move
            else:
                return best_score, best_move
        else:
            worst_score = float('inf')
            best_move = None
            valid_moves = reversi.board.get_valid_moves(reversi.board.current_player) 
            
            for move in valid_moves:
                new_board = deepcopy(reversi)
                new_board.play(move)
                
                score, candidate = self.minmax(
                    new_board,
                    depth + 1,
                    prev_move,
                    True
                )
                
                if score == float('inf'):
                    return score, candidate
                elif score < worst_score:
                    worst_score = score
                    best_move = candidate
                    
            if best_move is None:
                return worst_score, prev_move
            else:
                return worst_score, best_move

    
    def alpha_beta_pruning(self, reversi: Reversi, depth: int, prev_move: tuple[int, int], max_turn: bool, alpha: float, beta: float) -> tuple[float, tuple[int, int]]:
        if depth == self.max_depth or \
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
            max_eval = -float('inf')
            best_move = None
            valid_moves = reversi.board.get_valid_moves(reversi.board.current_player)
            
            for move in valid_moves:
                new_board = deepcopy(reversi)
                prev_move = move if depth == 0 else prev_move
                new_board.play(move)
                
                score, candidate = self.alpha_beta_pruning(
                    new_board,
                    depth + 1,
                    prev_move,
                    False,
                    alpha,
                    beta
                )
                
                if score == float('inf'):
                    return score, candidate
                elif score > max_eval:
                    max_eval = score
                    best_move = candidate
                    
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            if best_move is None:
                return max_eval, prev_move
            else:
                return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            valid_moves = reversi.board.get_valid_moves(reversi.board.current_player)
            
            for move in valid_moves:
                new_board = deepcopy(reversi)
                new_board.play(move)
                
                score, candidate = self.alpha_beta_pruning(
                    new_board,
                    depth + 1,
                    prev_move,
                    True,
                    alpha,
                    beta
                )
                
                if score == -float('inf'):
                    return score, candidate
                elif score < min_eval:
                    min_eval = score
                    best_move = candidate
                    
                beta = min(beta, score)
                if alpha >= beta:
                    break
                    
            if best_move is None:
                return min_eval, prev_move
            else:
                return min_eval, best_move
            