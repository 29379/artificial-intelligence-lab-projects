from typing import Optional, Union
from errors import *


class Coords:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        
    def __add__(self, elem: Union['Coords', tuple[int, int]]) -> 'Coords':
        if isinstance(elem, Coords):
            return Coords(self.x + elem.x, self.y + elem.y)
        elif isinstance(elem, tuple):
            return Coords(self.x + elem[0], self.y + elem[1])
        else:
            raise TypeError(f"Unsupported operand type for +: 'Coords' and '{type(elem)}'")
    
    def __eq__(self, elem: Union['Coords', tuple[int, int]]) -> bool:
        if isinstance(elem, Coords):
            return self.x == elem.x and self.y == elem.y
        elif isinstance(elem, tuple):
            return self.x == elem[0] and self.y == elem[1]
        else:
            raise TypeError(f"Unsupported operand type for ==: 'Coords' and '{type(elem)}'")

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
    
    def check_boundaries(self) -> bool:
        x_check = self.x >= 0 and self.x < 8
        y_check = self.y >= 0 and self.y < 8
        return x_check and y_check
    

class Board:
    PLAYERS = {'WHITE': 1, 'BLACK': 2}  

    def __init__(self) -> None:
        self.grid: dict[Coords, int] = {}   # 0 - empty, 1 - white, 2 - black
        self.current_player: int = self.PLAYERS.get('WHITE') # 1 - white, 2 - black
        self.directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        
        for x in range(8):
            for y in range(8):
                self.grid[Coords(x, y)] = None
        self.grid[Coords(3, 3)] = 1
        self.grid[Coords(4,4)] = 1
        self.grid[Coords(3,4)] = 2
        self.grid[Coords(4,3)] = 2
            
            
    def __str__(self) -> str:
        output = ""
        for y in range(8):
            for x in range(8):
                output += f"{self.grid.get(Coords(x, y))} "
            output += "\n"
        return output
    
    
    def flip(self, coords: Coords) -> None:
        if self.grid.get(coords) == 1:
            self.grid[coords] = 2
        elif self.grid.get(coords) == 2:
            self.grid[coords] = 1
        else:
            raise UnexpectedFieldStateError(f"Unexpected field state at: {str(coords)}: the state - {self.grid.get(coords)}")
        
        
    def get_valid_moves(self, start: Coords) -> list[Coords]:
        if not start.check_boundaries() or self.grid.get(start) != self.current_player:
            return []
        valid_moves: list[Coords] = []
        
        for unit_vector in self.directions:
            end = start + unit_vector
            if not end.check_boundaries() or self.grid.get(end) == self.current_player:
                continue
            if self.check_direction(start, end):
                valid_moves.append(end)
        return valid_moves
        
        
    def check_direction(self, start: Coords, unit_vector: Coords) -> bool:
        other_player = self.PLAYERS.get('BLACK') if self.current_player == self.PLAYERS.get('WHITE') else self.PLAYERS.get('WHITE')
        end = start + unit_vector
        
        if self.grid.get(end) != other_player or not end.check_boundaries():
            return False
        while self.grid.get(end) == other_player:
            end += unit_vector
            
        if self.grid.get(end) != 0 or not end.check_boundaries():
            return False
        return True
    
    
    # def is_valid_move(self, coords: Coords) -> bool:
    #     if self.grid.get(coords) != 0:
    #         return False
        
    #     for unit_vector in self.directions:
    #         if self.check_direction(coords, unit_vector):
    #             return True
    #     return False
