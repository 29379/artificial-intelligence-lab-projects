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


    def step(self, start: tuple[int, int], end: tuple[int, int], unit_vector: tuple[int, int]):
        x, y = start
        x2, y2 = end
        dx, dy = unit_vector
        if (x2 - x) * dy != (y2 - y) * dx:
            raise InvalidMoveError('The move direction is not valid')
        result = []
        while start != end:
            result.append(start)
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
        if self.grid[coords] == 1:
            self.grid[coords] = 2
        elif self.grid[coords] == 2:
            self.grid[coords] = 1
        else:
            raise UnexpectedFieldStateError(
                f"Unexpected field state at: {str(coords)}: the state - {self.grid[coords]}"
            )
        

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
        valid_moves = set()
        
        for x in range(8):
            for y in range(8):
                if self.grid[x, y] == player:
                    tmp = self.get_valid_moves_for_square((x, y), player)
                    valid_moves.update(tmp)
        return list(valid_moves)
                
        
    #   get all the moves that are possible for a player in that turn that originate in some coordinates
    def get_valid_moves_for_square(self, start: tuple[int, int], player: int) -> list[tuple[int, int]]:
        if not self.check_boundaries(start) or self.grid[start] != player:
            return []
        valid_moves: list[tuple[int, int]] = []
        
        for unit_vector in self.directions:
            end = start + unit_vector
            if not self.check_boundaries(end) or self.grid[end] == player:
                continue
            if self.check_direction(start, end):
                valid_moves.append(end)
        return valid_moves
        
        
    #   check whether I can make a move in some direction
    def check_direction(self, start: tuple[int, int], unit_vector: tuple[int, int], player: int) -> bool:
        enemy = self.BLACK if player == self.WHITE else self.WHITE
        end = np.add(start, unit_vector)
        
        #   check the logic here! maybe check what is in the place where I am right now after the while loop?
        if self.grid[end] != enemy or not self.check_boundaries(end):
            return False
        while self.grid[end] == enemy:
            end += unit_vector
            
        if self.grid[end] != self.EMPTY or not self.check_boundaries(end):
            return False
        return True
    
    
    def is_an_ally_field(self, coords: tuple[int, int]) -> bool:
        if self.check_boundaries(coords):
            return self.grid[coords] == self.current_player
        return False
    

    def is_an_empty_field(self, coords: tuple[int, int]) -> bool:
        if self.check_boundaries(coords):
            return self.grid[coords] == self.EMPTY
        return False
    
    
    def is_an_enemy_field(self, coords: tuple[int, int]) -> bool:
        if self.check_boundaries(coords):
            enemy_color = self.BLACK \
                if self.current_player == self.WHITE else self.WHITE
            return self.grid[coords] == enemy_color
        return False


    def count_pieces(self, player: int) -> int:
        if player != 1 and player != 2:
            raise UnexpectedPlayerStateError(
                f"Unexpected player - the value: {player}"
            )
        return np.count_nonzero(self.grid == player)


    #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    #   section dedicated to heuristic functions:

    def pieces_count_difference(self, player: int) -> float:
        if player != 1 and player != 2:
            raise UnexpectedPlayerStateError(
                f"Unexpected player - the value: {player}"
            )
        enemy = self.BLACK if player == self.WHITE else self.WHITE
        ally_pieces: int = np.count_nonzero(self.grid == player)
        enemy_pieces: int = np.count_nonzero(self.grid == enemy)
        return float(ally_pieces - enemy_pieces)


    def check_coin_parity_score(self, player: int) -> float:
        if player != 1 and player != 2:
            raise UnexpectedPlayerStateError(
                f"Unexpected player - the value: {player}"
            )
        enemy = self.BLACK if player == self.WHITE else self.WHITE
        ally_pieces: int = np.count_nonzero(self.grid == player)
        enemy_pieces: int = np.count_nonzero(self.grid == enemy)
        value = 0
        
        if ally_pieces + enemy_pieces != 0:
            value = 100 * (ally_pieces - enemy_pieces) / (ally_pieces + enemy_pieces)
        return value
    
    
    def check_mobility_score(self, player: int):
        if player != 1 and player != 2:
            raise UnexpectedPlayerStateError(
                f"Unexpected player - the value: {player}"
            )
        enemy = self.BLACK if player == self.WHITE else self.WHITE
        ally_moves = self.get_valid_moves(player)
        enemy_moves = self.get_valid_moves(enemy)
        value = 0
        
        if (ally_moves - enemy_moves != 0) and (ally_moves + enemy_moves != 0):
            value = 100 * (ally_moves - enemy_moves) / (ally_moves + enemy_moves)
        return value
        
        
    def check_corners_score(self, player: int):
        if player != 1 and player != 2:
            raise UnexpectedPlayerStateError(
                f"Unexpected player - the value: {player}"
            )
        enemy = self.BLACK if player == self.WHITE else self.WHITE
        ally_corners = 0
        enemy_corners = 0
        for corner in self.CORNERS.values():
            if self.grid[corner] == player:
                ally_corners += 1
            elif self.grid[corner] == enemy:
                enemy_corners += 1
            else:
                pass
        value = 0     
             
        if (ally_corners - enemy_corners != 0) and (ally_corners + enemy_corners != 0):            
            value = 100 * (ally_corners - enemy_corners) / (ally_corners + enemy_corners) 
        return value
        
    
    def check_stability_score(self, player: int):
        if player != 1 and player != 2:
            raise UnexpectedPlayerStateError(
                f"Unexpected player - the value: {player}"
            )
        enemy = self.BLACK if player == self.WHITE else self.WHITE
        ally_stability_count = 0
        enemy_stability_count = 0
        
        for x in range(8):
            for y in range(8):
                if self.grid[x, y] == player:
                    is_stable = self.check_piece_stability((x, y))
                    if is_stable:
                        ally_stability_count += 1
                    elif self.grid[x, y] == enemy:
                        is_stable = self.check_piece_stability((x, y))
                        if is_stable:
                            enemy_stability_count += 1

        value = 0
        if (ally_stability_count - enemy_stability_count != 0) and (ally_stability_count + enemy_stability_count != 0):
            value = 100 * (ally_stability_count - enemy_stability_count) / (ally_stability_count + enemy_stability_count)      
        return value   
    
    
    def is_stable(self, coords: tuple[int, int]) -> bool:
        x, y = coords
        player = self.grid[x, y]
        if player != 1 and player != 2:
            raise UnexpectedPlayerStateError(
                f"Unexpected player - the value: {player}"
        )
        
        #   loop through all directions
        for unit_vector in self.directions:
            dx, dy = unit_vector
            ally_fields = 0
            empty_fields = 0
            #   x2 and y2 help me look for stable lines in a particular direction
            x2 = x + dx
            y2 = y + dy
            
            #   go as far as possible in a particular direction
            while (0 <= x2 < 8) and (0 <= y2 < 8):
                if self.grid[x2, y2] == player:
                    ally_fields += 1
                elif self.grid[x2, y2] == self.EMPTY:
                    empty_fields += 1
                else:
                    break
                x2 += dx
                y2 += dy
                
            #   if the line ends with an ally piece, it is stable
            if self.is_an_ally_field((x2, y2)):
                continue
            
            #   if the line ends with an empty field, it is stable 
            #   IF there are no allies on any other side of the line
            if ally_fields == 0 or \
                    (x - dx >= 0) and (self.grid[x-dx, y-dy] == player) or \
                    (x2 + dx < 8) and self.grid[x2+dx, y2+dy] == player:
                continue
            
            #   if i managed to get here, the line is unstable
            return False
        #   if all the lines are stable, the piece is stable as well
        return True
                
    
    #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    #   remember to update number of heuristics when new stuf gets written!!!!
    def check_game_state(self, player: int) -> float:
        number_of_heuristics = 5
        #   average game state value
        game_state = (self.pieces_count_difference(player) + \
            self.check_coin_parity_score(player) + \
            self.check_corners_score(player) + \
            self.check_mobility_score(player) + \
            self.check_stability_score(player)) / number_of_heuristics
        return game_state
