from algorithms import *
from reversi import *


class Player:
    WHITE = 1
    BLACK = 2
    
    def __init__(self, field: int, depth: int, heuristic) -> None:
        self.field: int = field
        self.alg: Algorithms = Algorithms(depth, heuristic)
        
        
    def make_a_move_minmax(self, reversi: Reversi) -> tuple[float, tuple[int, int]]:
        return self.alg.minmax(reversi, 0, None, True)


    def make_a_move_alpha_beta(self, reversi: Reversi) -> tuple[float, tuple[int, int]]:
        return self.alg.alpha_beta_pruning(reversi, 0, None, True, -float('inf'), float('inf'))
        
    