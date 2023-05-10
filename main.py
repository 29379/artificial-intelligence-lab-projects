from algorithms import *
from reversi import *
from board import *
from player import *
from errors import *
from heuristics import *

from datetime import datetime
from copy import deepcopy4
import csv, itertools
    
    
depths = [(1, 1), (1, 2), (1, 3),
          (2, 1), (2, 2), (2, 3), 
          (3, 1), (3, 2), (3, 3)]

heuristics = [stability, corners, coin_parity, mobility]

heuristic_names = {}


def minmax_test(reversi: Reversi, white_player: Player, black_player: Player):
    start_time = datetime.now()
    end_time = datetime.now()
    while reversi.game_status == reversi.GAME_STATUS.get('IN_PROGRESS'):
        if reversi.board.current_player == white_player.field:
            _, field = white_player.make_a_move_minmax(deepcopy(reversi))
            reversi.play(field)
        else: 
            _, field = black_player.make_a_move_minmax(deepcopy(reversi))
            reversi.play(field)
    runtime = (end_time - start_time).total_seconds()
    
    return reversi.game_status, \
       reversi.board.count_pieces(1), \
       reversi.board.count_pieces(2), \
       runtime, \
       False
       

def alpha_beta_test(reversi: Reversi, white_player: Player, black_player: Player):
    start_time = datetime.now()
    end_time = datetime.now()
    while reversi.game_status == reversi.GAME_STATUS.get('IN_PROGRESS'):
        if reversi.board.current_player == white_player.field:
            _, field = white_player.make_a_move_alpha_beta(deepcopy(reversi))
            reversi.play(field)
        else: 
            _, field = black_player.make_a_move_alpha_beta(deepcopy(reversi))
            reversi.play(field)
    runtime = (end_time - start_time).total_seconds()
    
    return reversi.game_status, \
       reversi.board.count_pieces(1), \
       reversi.board.count_pieces(2), \
       runtime, \
       False


def write_solution_to_file(self, file_name, rows):
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["Game result",
                         "Winner score",
                         "Loser score"
                         "Runtime",
                         "White player heuristic",
                         "Black player heuristic",
                         "Min-max depth",
                         "Alpha-beta-pruning"
                         ])
        for row in rows:
            writer.writerow([row['game_result'],
                    row['winner_score'],
                    row['loser_score'],
                    row['runtime'],
                    row['heuristic1_name'],
                    row['heuristic2_name'],
                    row['depth1'],
                    row['depth2'],
                    row['alpha_beta']])
    

def main():
    minmax_results = []
    alpha_beta_results = []
    for depth1, depth2 in itertools.product(depths, repeat=2):
        for heuristic1, heuristic2 in itertools.product(heuristics, repeat=2):
            player1 = Player(1, depth1, heuristic1)
            player2 = Player(2, depth2, heuristic2)
            reversi = Reversi()
            
            game_result, winner_score, loser_score, runtime, alpha_beta = minmax_test(deepcopy(reversi), deepcopy(player1), deepcopy(player2))
            minmax_results.append({
                'game_result': game_result, 
                'winner_score': winner_score, 
                'loser_score': loser_score, 
                'runtime': runtime, 
                'heuristic1_name': heuristic_names.get(heuristic1), 
                'heuristic2_name': heuristic_names.get(heuristic2), 
                'depth1': depth1, 
                'depth2': depth2, 
                'alpha_beta': alpha_beta
            })
            
            game_result, winner_score, loser_score, runtime, alpha_beta = alpha_beta_test(deepcopy(reversi), deepcopy(player1), deepcopy(player2))
            alpha_beta_results.append({
                'game_result': game_result, 
                'winner_score': winner_score, 
                'loser_score': loser_score, 
                'runtime': runtime, 
                'heuristic1_name': heuristic_names.get(heuristic1), 
                'heuristic2_name': heuristic_names.get(heuristic2), 
                'depth1': depth1, 
                'depth2': depth2, 
                'alpha_beta': alpha_beta
            })
    
    minmax_results = sorted(minmax_results, key=lambda x: x['runtime'])
    alpha_beta_results = sorted(alpha_beta_results, key=lambda x: x['runtime'])
    
    write_solution_to_file('min-max.csv', minmax_results)
    write_solution_to_file('alpha-beta.csv', alpha_beta_results)
    
    print('\nMIN-MAX RESULTS\n')
    
    for (game_result, heuristic1, heuristic2), group in itertools.groupby(minmax_results, key=lambda x: (x['game_result'], x['heuristic1_name'], x['heuristic2_name'])):
        runtime = [x['runtime'] for x in group]
        print(f"{game_result} - {runtime} | {heuristic1} - {heuristic2}")

        
    print('\nALPHA-BETA-PRUNING RESULTS\n')
        
    for (game_result, heuristic1, heuristic2), group in itertools.groupby(alpha_beta_results, key=lambda x: (x['game_result'], x['heuristic1_name'], x['heuristic2_name'])):
        runtime = [x['runtime'] for x in group]
        print(f"{game_result} - {runtime} | {heuristic1} - {heuristic2}")



if __name__ == '__main__':
    main()
    