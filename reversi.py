from board import *
from errors import *
from player import *

class Reversi:
    EMPTY = 0
    WHITE = 1
    BLACK = 2
    
    GAME_STATES = {
        "IN_PROGRESS": "In progress",
        "WHITE_WIN": "White pieces win",
        "BLACK_WIN": "Black pieces win",
        "TIE": "It's a tie"
    }
    
    def __init__(self) -> None:
        self.board = Board()
        self.white_player = Player(self.WHITE)
        self.black_player = Player(self.BLACK)
        
        
    def play(self, coord):
        pass
        