from board import *
from errors import *


class Reversi:
    GAME_STATUS = {
        "IN_PROGRESS": "Game in progress",
        "WHITE_WIN": "White pieces win",
        "BLACK_WIN": "Black pieces win",
        "TIE": "It's a tie"
    }
    
    def __init__(self) -> None:
        self.board: Board = Board()
        self.game_mode: str = None
        self.game_status: str = self.GAME_STATUS.get('IN_PROGRESS')
        self.rounds = 0
    
    
    def check_game_status(self) -> str:
        return self.game_status
    
    
    def play(self, coord: tuple[int, int]) -> 'Reversi':
        if self.game_status != self.GAME_STATUS.get('IN_PROGRESS'):
            raise GameHasEndedError('The game is over already')
        #   there are no viable moves left for the current player
        if coord is None:
            self.game_status = self.finalize_game()
            return
        if coord not in self.board.get_valid_moves(coord):
            raise InvalidMoveError(f"The move to {coord} is not valid here")
        
        fields = []
        for unit_vector in self.board.directions:
            curr = coord + unit_vector
            while self.board.is_an_enemy_field(tuple(curr)):
                curr += unit_vector
            if self.board.is_an_ally_field(tuple(curr)):
                fields += self.board.step(coord, curr, unit_vector)

        #   finalize the move
        for field in fields:
            self.board.flip(tuple(field))
        self.board.change_current_player()
        self.rounds += 1
        if self.rounds >= 62:
            self.game_status = self.finalize_game()
        
        return self
    
    
    def finalize_game(self) -> str:
        if not self.board.get_valid_moves(self.board.current_player):
            self.board.change_current_player()
            if not self.board.get_valid_moves(self.board.current_player):
                white = self.board.count_pieces(self.board.WHITE)
                black = self.board.count_pieces(self.board.BLACK)
                if white > black:
                    return self.GAME_STATUS.get('WHITE_WIN')
                elif black > white:
                    return self.GAME_STATUS.get('BLACK_WIN')
                else:
                    return self.GAME_STATUS.get('TIE')
        return self.GAME_STATUS.get('IN_PROGRESS')
        