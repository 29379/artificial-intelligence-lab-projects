from typing import Optional, Union
from errors import *
import numpy as np

class Board:
    PLAYERS = {'WHITE': 1, 'BLACK': 2}  
    EMPTY = 0
    WHITE = 1
    BLACK = 2
    
    def __init__(self) -> None:
        self.grid: np.ndarray = np.zeros((8, 8), dtype=np.int8)   # 0 - empty, 1 - white, 2 - black
        self.current_player: int = self.PLAYERS.get('WHITE') # 1 - white, 2 - black
        self.directions: np.ndarray = np.array([(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)], dtype=np.int8)
        
        self.grid[3, 3] = self.WHITE
        self.grid[4, 4] = self.WHITE
        self.grid[3, 4] = self.BLACK
        self.grid[4, 3] = self.BLACK
            
            
    def __str__(self) -> str:
        return str(self.grid)


    def check_boundaries(self, coords: tuple[int, int]) -> bool:
        x_check = coords[0] >= 0 and coords[0] < 8
        y_check = coords[1] >= 0 and coords[1] < 8
        return x_check and y_check


    def flip(self, coords: tuple[int, int]) -> None:
        if self.grid[coords] == 1:
            self.grid[coords] = 2
        elif self.grid[coords] == 2:
            self.grid[coords] = 1
        else:
            raise UnexpectedFieldStateError(f"Unexpected field state at: {str(coords)}: the state - {self.grid[coords]}")
        

    def get_valid_moves(self) -> list[tuple[int, int]]:
        valid_moves = set()
        
        for x in range(8):
            for y in range(8):
                if self.grid[x, y] == self.current_player:
                    tmp = self.get_valid_moves_for_square((x, y))
                    valid_moves.update(tmp)
        return list(valid_moves)
                
        
        
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
        
        
    def check_direction(self, start: tuple[int, int], unit_vector: tuple[int, int]) -> bool:
        other_player = self.PLAYERS.get('BLACK') if self.current_player == self.PLAYERS.get('WHITE') else self.PLAYERS.get('WHITE')
        end = np.add(start, unit_vector)
        
        if self.grid[end] != other_player or not self.check_boundaries(end):
            return False
        while self.grid[end] == other_player:
            end += unit_vector
            
        if self.grid[end] != 0 or not self.check_boundaries(end):
            return False
        return True
    
    
    def is_an_ally_field(self, coords: tuple[int, int]) -> bool:
        return self.check_boundaries(coords) and self.grid[coords] == self.current_player


    def is_an_empty_field(self, coords: tuple[int, int]) -> bool:
        return self.check_boundaries(coords) and self.grid[coords] == 0
    

    def count_pieces(self) -> int:
        white_pieces: int = np.count_nonzero(self.grid == 1)
        black_pieces: int = np.count_nonzero(self.grid == 2)
        return white_pieces - black_pieces
