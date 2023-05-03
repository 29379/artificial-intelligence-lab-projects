from typing import Optional, Union
from errors import *
from player import *
import numpy as np

class Board:
    CORNERS = {'UPPER_LEFT': (0, 0), 'UPPER_RIGHT': (0, 7),
               'LOWER_LEFT': (7, 0), 'LOWER_RIGHT': (7, 7)}
    EMPTY = 0
    WHITE = 1
    BLACK = 2
    
    def __init__(self) -> None:
        self.grid: np.ndarray = np.zeros((8, 8), dtype=np.int8)   # 0 - empty, 1 - white, 2 - black
        self.current_player: int = self.WHITE # 1 - white, 2 - black
        self.directions: np.ndarray = np.array(
            [(0, -1), (0, 1), (-1, 0), (1, 0),
             (-1, -1), (1, 1), (-1, 1), (1, -1)],
            dtype=np.int8)
        
        self.grid[3, 3] = self.WHITE
        self.grid[4, 4] = self.WHITE
        self.grid[3, 4] = self.BLACK
        self.grid[4, 3] = self.BLACK
            
            
    def __str__(self) -> str:
        return str(self.grid)


    def change_current_player(self) -> None:
        self.current_player = self.BLACK \
            if self.current_player == self.WHITE else self.WHITE


    #   check if I dont step out of the board
    def check_boundaries(self, coords: tuple[int, int]) -> bool:
        x_check = coords[0] >= 0 and coords[0] < 8
        y_check = coords[1] >= 0 and coords[1] < 8
        return x_check and y_check


    #   flip a piece located on some coordinates
    def flip(self, coords: tuple[int, int]) -> None:
        if self.grid[coords] == 1:
            self.grid[coords] = 2
        elif self.grid[coords] == 2:
            self.grid[coords] = 1
        else:
            raise UnexpectedFieldStateError(
                f"Unexpected field state at: {str(coords)}: the state - {self.grid[coords]}"
            )
        

    #   get the coordinates of all the ally pieces for a current player
    def get_player_pieces(self) -> list[tuple[int, int]]:
        coords: list[tuple[int, int]] = []
        for x in range(8):
            for y in range(8):
                if self.grid[x, y] == self.current_player:
                    coords.append([x, y])
        return coords


    #   get all the moves that are possible for a player in that 
    #   make it take a player parameter
    def get_valid_moves(self, player: Player) -> list[tuple[int, int]]:
        valid_moves = set()
        
        for x in range(8):
            for y in range(8):
                if self.grid[x, y] == self.current_player:
                    tmp = self.get_valid_moves_for_square((x, y))
                    valid_moves.update(tmp)
        return list(valid_moves)
                
        
    #   get all the moves that are possible for a player in that turn that originate in some coordinates
    def get_valid_moves_for_square(self, start: tuple[int, int]) -> list[tuple[int, int]]:
        if not self.check_boundaries(start) or self.grid[start] != self.current_player:
            return []
        valid_moves: list[tuple[int, int]] = []
        
        for unit_vector in self.directions:
            end = start + unit_vector
            if not self.check_boundaries(end) or self.grid[end] == self.current_player:
                continue
            if self.check_direction(start, end):
                valid_moves.append(end)
        return valid_moves
        
        
    #   check whether I can make a move in some direction
    def check_direction(self, start: tuple[int, int], unit_vector: tuple[int, int]) -> bool:
        other_player = self.BLACK if self.current_player == self.WHITE else self.WHITE
        end = np.add(start, unit_vector)
        
        #   check the logic here! maybe check what is in the place where I am right now after the while loop?
        if self.grid[end] != other_player or not self.check_boundaries(end):
            return False
        while self.grid[end] == other_player:
            end += unit_vector
            
        if self.grid[end] != 0 or not self.check_boundaries(end):
            return False
        return True
    
    
    def is_an_ally_field(self, coords: tuple[int, int]) -> bool:
        if self.check_boundaries(coords):
            return self.grid[coords] == self.current_player
        return False
    

    def is_an_empty_field(self, coords: tuple[int, int]) -> bool:
        if self.check_boundaries(coords):
            return self.grid[coords] == 0
        return False
    
    
    def is_an_enemy_field(self, coords: tuple[int, int]) -> bool:
        if self.check_boundaries(coords):
            enemy_color = self.BLACK \
                if self.current_player == self.WHITE else self.WHITE
            return self.grid[coords] == enemy_color
        return False


    def count_pieces(self) -> float:
        white_pieces: int = np.count_nonzero(self.grid == self.WHITE)
        black_pieces: int = np.count_nonzero(self.grid == self.BLACK)
        return float(white_pieces - black_pieces)


    #   heuristic functions:
    def check_coin_parity(self) -> float:
        white_pieces: int = np.count_nonzero(self.grid == self.WHITE)
        black_pieces: int = np.count_nonzero(self.grid == self.BLACK)
        value = 100 * (white_pieces - black_pieces) / (white_pieces + black_pieces)
        
        if self.current_player == self.WHITE:
            return value
        elif self.current_player == self.BLACK:
            return (100 - value)
        else:
            raise UnexpectedPlayerStateError(f"Unexpected current player - the value: {self.current_player}")
    
    #   change it so it takes a player as a parameter
    def check_mobility(self):
        white_moves = self.get_valid_moves('WHITE')
        black_moves = self.get_valid_moves('BLACK')
        value = 0
        
        if white_moves - black_moves != 0:
            value = 100 * (white_moves - black_moves) / (white_moves + black_moves)
            if self.current_player == self.WHITE:
                return value
            elif self.current_player == self.BLACK:
                return (100 - value)
            else:
                raise UnexpectedPlayerStateError(f"Unexpected current player - the value: {self.current_player}")
        else:
            return value    
    
    def check_corners(self):
        white_corners = 0
        black_corners = 0
        for corner in self.CORNERS.values():
            if self.grid[corner] == self.WHITE:
                white_corners += 1
            elif self.grid[corner] == self.BLACK:
                black_corners += 1
            else:
                pass
             
        if white_corners - black_corners != 0:            
            value = 100 * (white_corners - black_corners) / (white_corners + black_corners) 
            if self.current_player == self.WHITE:
                return value
            else:
                return (100 - value)
        else:
            return value
        
    
    #   wtf????
    def check_stability(self):
        return 0
    
    #   remember to update that when new stuf gets written!!!!!!!!!!!!!!!!!
    def check_game_state(self) -> float:
        number_of_heuristics = 5
        game_state = (self.count_pieces + \
            self.check_coin_parity() + \
            self.check_corners + \
            self.check_mobility + \
            self.check_stability) / number_of_heuristics
        return game_state