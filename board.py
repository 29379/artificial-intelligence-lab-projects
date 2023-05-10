from typing import Optional, Union
from errors import *
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


    def step(self, start: tuple[int, int], end: tuple[int, int], unit_vector: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = start
        x2, y2 = end
        dx, dy = unit_vector
        if (x2 - x) * dy != (y2 - y) * dx:
            raise InvalidMoveError('The move direction is not valid')
        result = []
        while tuple(start) != tuple(end):
            result.append(tuple(start))
            start += unit_vector
        return result


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
        # if self.grid[coords] == 1:
        #     self.grid[coords] = 2
        # elif self.grid[coords] == 2:
        #     self.grid[coords] = 1
        # elif self.grid[coords] == 0:
        #     pass
        # else:
        #     raise UnexpectedFieldStateError(
        #         f"Unexpected field state at: {str(coords)}: the state - {self.grid[coords]}"
        #     )
        if self.grid[coords] != self.current_player:
            self.grid[coords] = self.current_player

    #   get the coordinates of all the ally pieces for a current player
    def get_player_pieces(self, player: int) -> list[tuple[int, int]]:
        coords: list[tuple[int, int]] = []
        for x in range(8):
            for y in range(8):
                if self.grid[x, y] == player:
                    coords.append([x, y])
        return coords


    #   get all the moves that are possible for a specific player
    def get_valid_moves(self, player: int) -> list[tuple[int, int]]:
        valid_moves: list[tuple[int, int]] = []
        pieces = self.get_player_pieces(self.current_player)
        # for x in range(8):
        #     for y in range(8):
        #         if self.grid[x, y] == player:
        #             tmp = self.get_valid_moves_for_square((x, y), player)
        #             valid_moves.update(tmp)
        for piece in pieces:
            for unit_vector in self.directions:
                end = piece + unit_vector
                while self.is_an_enemy_field(tuple(end)):
                    end += unit_vector
                    if self.is_an_empty_field(tuple(end)):
                        if tuple(end) not in valid_moves:
                            valid_moves.append(tuple(end))
        return valid_moves
                
        
    #   get all the moves that are possible for a player in that turn that originate in some coordinates
    # def get_valid_moves_for_square(self, start: tuple[int, int], player: int) -> list[tuple[int, int]]:
    #     if not self.check_boundaries(start) or self.grid[start] != player:
    #         return []
    #     valid_moves: list[tuple[int, int]] = []
        
    #     for unit_vector in self.directions:
    #         end = start + unit_vector
    #         if not self.check_boundaries(end) or np.any(self.grid[end] == player):
    #             continue
    #         if self.check_direction(start, end):
    #             valid_moves.append(end)
    #     return valid_moves
        
        
    # def check_direction(self, start: tuple[int, int], unit_vector: tuple[int, int]) -> bool:
    #     enemy = self.BLACK if self.current_player == self.WHITE else self.WHITE
    #     end = np.add(start, unit_vector)
        
    #     #   check the logic here! maybe check what is in the place where I am right now after the while loop?
    #     if not self.check_boundaries(end) or np.any(self.grid[end] != enemy):
    #         return False
    #     while np.any(self.grid[end] == enemy):
    #         end += unit_vector
            
    #     if not self.check_boundaries(end) or np.any(self.grid[end] != self.EMPTY):
    #         return False
    #     return True
    
    
    def is_an_ally_field(self, coords: tuple[int, int]) -> bool:
        if self.check_boundaries(coords):
            return np.all(self.grid[coords] == self.current_player)
        return False
    

    def is_an_empty_field(self, coords: tuple[int, int]) -> bool:
        if self.check_boundaries(coords):
            return np.all(self.grid[coords] == self.EMPTY)
        return False
    
    
    def is_an_enemy_field(self, coords: tuple[int, int]) -> bool:
        if self.check_boundaries(coords):
            enemy_color = self.BLACK \
                if self.current_player == self.WHITE else self.WHITE
            return np.all(self.grid[coords] == enemy_color)
        return False


    def count_pieces(self, player: int) -> int:
        if player != 1 and player != 2:
            raise UnexpectedPlayerStateError(
                f"Unexpected player - the value: {player}"
            )
        return np.count_nonzero(self.grid == player)


    def pieces_count_difference(self, player: int) -> float:
        if player != 1 and player != 2:
            raise UnexpectedPlayerStateError(
                f"Unexpected player - the value: {player}"
            )
        enemy = self.BLACK if player == self.WHITE else self.WHITE
        ally_pieces: int = np.count_nonzero(self.grid == player)
        enemy_pieces: int = np.count_nonzero(self.grid == enemy)
        return float(ally_pieces - enemy_pieces)
