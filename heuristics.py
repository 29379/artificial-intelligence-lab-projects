from reversi import *
from errors import *
from board import *

# def stability(game: Reversi, player: int) -> float:
#     board = game.board.grid
#     enemy = 1 if player == 2 else 2

#     ally_stable_tiles = 0
#     enemy_stable_tiles = 0
    
#     for i in range(8):
#         for j in range(8):
#             if board[i, j] == player:
#                 if is_stable((i, j), game.board, player):
#                     ally_stable_tiles += 1
#             elif board[i, j] == enemy:
#                 if is_stable(i, j, game.board, enemy):
#                     enemy_stable_tiles += 1

#     if ally_stable_tiles > enemy_stable_tiles:
#         result = (100.0 * ally_stable_tiles) / (ally_stable_tiles + enemy_stable_tiles)
#     elif ally_stable_tiles < enemy_stable_tiles:
#         result = -(100.0 * enemy_stable_tiles) / (ally_stable_tiles + enemy_stable_tiles)
#     else:
#         result = 0

#     return result

def stability(game: Reversi) -> float:
    board = game.board.grid
    if game.board.current_player != 1 and game.board.current_player != 2:
        raise UnexpectedPlayerStateError(
            f"Unexpected player - the value: {game.board.current_player}"
        )
    enemy = game.board.BLACK if game.board.current_player == game.board.WHITE else game.board.WHITE

    ally_stable_tiles = 0
    enemy_stable_tiles = 0

    for i in range(8):
        for j in range(8):
            if board[i][j] == game.board.current_player:
                if is_stable((i, j), game.board, game.board.current_player):
                    ally_stable_tiles += 1
            elif board[i][j] == enemy:
                if is_stable((i, j), game.board, enemy):
                    enemy_stable_tiles += 1

    total_stable_tiles = ally_stable_tiles + enemy_stable_tiles

    #   return 50 if there areno stable tiles
    if total_stable_tiles == 0:
        return 50.0 

    ally_stability = (100.0 * ally_stable_tiles) / total_stable_tiles
    enemy_stability = (100.0 * enemy_stable_tiles) / total_stable_tiles

     #  scale output to 0-100 range
    if ally_stability > enemy_stability:
        result = ally_stability / 2 
    else:
        result = -enemy_stability / 2 

    result += 50  # shift output to 0-100 range

    return result



def is_stable(coords: tuple[int, int], board: Board, color: int) -> bool:
        x, y = coords
        player = board.grid[x, y]
        if player != 1 and player != 2:
            raise UnexpectedPlayerStateError(
                f"Unexpected player - the value: {player}"
        )
        
        #   loop through all directions
        for unit_vector in board.directions:
            dx, dy = unit_vector
            ally_fields = 0
            empty_fields = 0
            #   x2 and y2 help me look for stable lines in a particular direction
            x2 = x + dx
            y2 = y + dy
            
            #   go as far as possible in a particular direction
            while (0 <= x2 < 8) and (0 <= y2 < 8):
                if board.grid[x2, y2] == player:
                    ally_fields += 1
                elif board.grid[x2, y2] == board.EMPTY:
                    empty_fields += 1
                else:
                    break
                x2 += dx
                y2 += dy
                
            #   if the line ends with an ally piece, it is stable
            if board.is_an_ally_field((x2, y2)):
                continue
            
            #   if the line ends with an empty field, it is stable 
            #   IF there are no allies on any other side of the line
            if ally_fields == 0 or \
                    (0 <= x - dx < 8) and (0 <= y - dy < 8) and (board.grid[x-dx, y-dy] == player) or \
                    (0 <= x2 + dx < 8) and (0 <= y2 + dy < 8) and board.grid[x2+dx, y2+dy] == player:
                continue
            
            #   if i managed to get here, the line is unstable
            return False
        #   if all the lines are stable, the piece is stable as well
        return True
    

#   ------------------------------------------------------------------------------

def corners(game: Reversi) -> float:
    board = game.board.grid
    if game.board.current_player != 1 and game.board.current_player != 2:
        raise UnexpectedPlayerStateError(
        f"Unexpected player - the value: {game.board.current_player}"
    )
    enemy = game.board.BLACK if game.board.current_player == game.board.WHITE else game.board.WHITE

    V = [
        [20, -3, 11, 8, 8, 11, -3, 20],
        [-3, -7, -4, 1, 1, -4, -7, -3],
        [11, -4, 2, 2, 2, 2, -4, 11],
        [8, 1, 2, -3, -3, 2, 1, 8],
        [8, 1, 2, -3, -3, 2, 1, 8],
        [11, -4, 2, 2, 2, 2, -4, 11],
        [-3, -7, -4, 1, 1, -4, -7, -3],
        [20, -3, 11, 8, 8, 11, -3, 20]
    ]

    total_weight = 0

    for i in range(8):
        for j in range(8):
            if board[i][j] == game.board.current_player:
                total_weight += V[i][j]
            elif board[i][j] == enemy:
                total_weight -= V[i][j]

    scaled_output = (total_weight + 288) / 5.76
    return scaled_output

#   ------------------------------------------------------------------------------

def coin_parity(game: Reversi) -> float:
    if game.board.current_player != 1 and game.board.current_player != 2:
        raise UnexpectedPlayerStateError(
            f"Unexpected player - the value: {game.board.current_player}"
        )
    enemy = game.board.BLACK if game.board.current_player == game.board.WHITE else game.board.WHITE
    ally_pieces: int = np.count_nonzero(game.board.grid == game.board.current_player)
    enemy_pieces: int = np.count_nonzero(game.board.grid == enemy)
    value = 0
    
    if ally_pieces + enemy_pieces != 0:
        value = 100 * (ally_pieces - enemy_pieces) / (ally_pieces + enemy_pieces)
    return value

#   ------------------------------------------------------------------------------

def mobility(game: Reversi):
    if game.board.current_player != 1 and game.board.current_player != 2:
        raise UnexpectedPlayerStateError(
            f"Unexpected player - the value: {game.board.current_player}"
        )
    enemy = game.board.BLACK if game.board.current_player == game.board.WHITE else game.board.WHITE
    ally_moves = game.board.get_valid_moves(game.board.current_player)
    enemy_moves = game.board.get_valid_moves(enemy)
    value = 0
    
    if (len(ally_moves) - len(enemy_moves) != 0) and (len(ally_moves) + len(enemy_moves) != 0):
        value = 100 * (len(ally_moves) - len(enemy_moves)) / (len(ally_moves) + len(enemy_moves))
    return value

#   ------------------------------------------------------------------------------