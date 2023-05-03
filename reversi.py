from board import *
from errors import *

class Reversi:
    GAME_STATUS = {
        "IN_PROGRESS": "Game in progress",
        "WHITE_WIN": "White pieces win",
        "BLACK_WIN": "Black pieces win",
        "TIE": "It's a tie"
    }
    GAME_MODES = {
        "PLAYER_VS_PLAYER": "Player vs Player",
        "CPU_VS_CPU": "CPU vs CPU",
        "PLAYER_VS_CPU": "Player vs CPU"
    }
    
    def __init__(self) -> None:
        self.board: Board = Board()
        self.game_mode: str = None
        self.game_status: str = None
        self.rounds = 0
        self.white_score = None
        self.black_score = None
    
    
    def check_game_status(self):
        return self.game_status
    
    
    def play(self, coord: tuple[int, int]) -> 'Reversi':
        if self.game_status != self.GAME_STATUS.get('IN_PROGRESS'):
            raise GameHasEndedError('The game is over already')
        #   there are no viable moves left for the current player
        if coord is None:
            self.game_status = self.check_endgame()
            return
        if coord not in self.board.get_valid_moves(coord):
            raise InvalidMoveError(f"The move to {coord} is not valid here")
        
        fields = []
        for unit_vector in self.board.directions:
            curr = coord + unit_vector
            while self.board.is_an_enemy_field(curr):
                curr += unit_vector
            if self.board.is_an_ally_field(curr):
                fields += self.board.step(coord, curr, unit_vector)

        #   finalize the move
        for field in fields:
            self.board.flip(field)
        self.board.change_current_player()
        self.rounds += 1
        self.white_score = self.board.check_game_state(self.board.WHITE)
        self.black_score = self.board.check_game_state(self.board.BLACK)
        if self.rounds >= 62:
            self.game_status = self.finalize_game()
        
        #   necessary for method chaining
        return self
    
    
    def finalize_game(self) -> str:
        if not self.board.get_valid_moves(self.board.current_player):
            self.board.change_current_player()
            if not self.board.get_valid_movese(self.board.current_player):
                white = self.board.count_pieces(self.board.WHITE)
                black = self.board.count_pieces(self.board.BLACK)
                if white > black:
                    return self.GAME_STATUS.get('WHITE_WIN')
                elif black > white:
                    return self.GAME_STATUS.get('BLACK_WIN')
                else:
                    return self.GAME_STATUS.get('TIE')
        return self.GAME_STATUS.get('IN_PROGRESS')
        