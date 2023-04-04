from DataReader import DataReader
from graph import *
from Algorithms import Algorithms
from datetime import datetime, timedelta
import random, csv
import numpy as np


def main() -> None:
    graph = DataReader.load_data()
    starting_nodes, ending_nodes, starting_times = get_samples(graph)
    with open('output.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Stop', 'Runtime', 'Line', 'Departure time', 'Arrival time'])
    
    dijkstra_sample100(graph, starting_nodes, ending_nodes, starting_times)
    astar_time_sample100(graph, starting_nodes, ending_nodes, starting_times)
    astar_transfers_sample100(graph, starting_nodes, ending_nodes, starting_times)
    
    #   dijkstra_single(graph, "Psie Pole", "FAT", timedelta(hours=14, minutes=33))
    #   astar_time_single(graph, "Psie Pole", "FAT", timedelta(hours=14, minutes=33))
    #   astar_transfers_single(graph, "Psie Pole", "FAT", timedelta(hours=14, minutes=33))
    
    # astar_time_single(graph, "Rynek", "Zamkowa", timedelta(hours=14, minutes=33))
    # astar_time_single(graph, "Rynek", "PL. GRUNWALDZKI", timedelta(hours=14, minutes=33))
    # astar_time_single(graph, "MOKRY DWOR", "Dolmed", timedelta(hours=14, minutes=33))
    # astar_time_single(graph, "MOKRY DWOR", "Dolmed", timedelta(hours=19, minutes=33))
    # astar_time_single(graph, "MOKRY DWOR", "Dolmed", timedelta(hours=6, minutes=33))
    # astar_time_single(graph, "MOKRY DWOR", "Dolmed", timedelta(hours=0, minutes=33))
    
    
    # astar_transfers_single(graph, "Rynek", "Zamkowa", timedelta(hours=14, minutes=33))
    # astar_transfers_single(graph, "Rynek", "PL. GRUNWALDZKI", timedelta(hours=14, minutes=33))
    # astar_transfers_single(graph, "MOKRY DWOR", "Dolmed", timedelta(hours=14, minutes=33))
    # astar_transfers_single(graph, "MOKRY DWOR", "Dolmed", timedelta(hours=19, minutes=33))
    # astar_transfers_single(graph, "MOKRY DWOR", "Dolmed", timedelta(hours=6, minutes=33))
    # astar_transfers_single(graph, "MOKRY DWOR", "Dolmed", timedelta(hours=0, minutes=33))
    
    # astar_transfers_single(graph, "Rozanka", "GAJ", timedelta(hours=4, minutes=18))
    # astar_transfers_single(graph, "Smocza", "Kielczowska", timedelta(hours=3, minutes=38))
    #astar_transfers_single(graph, "Warminska", "Kielczow - WODROL", timedelta(hours=21, minutes=19))

def trim_extreme_cases(runtimes: list[float]) -> list[float]:
    n_trim = int(0.05 * len(runtimes))
    for i in range(n_trim):
        max_runtime = max(runtimes)
        runtimes.remove(max_runtime)
        min_runtime = min(runtimes)
        runtimes.remove(min_runtime)
    return runtimes


def get_samples(graph: Graph) -> tuple[list[str], list[str], list[timedelta]]:
    starting_nodes = random.sample(list(graph.nodes.keys()), 5)
    ending_nodes = random.sample(list(graph.nodes.keys()), 5)
    starting_times = []
    
    for i in range(len(starting_nodes)):
        hours = random.randint(0, 23)
        minutes = random.randint(1, 59)
        starting_times.append(timedelta(hours=hours, minutes=minutes))     
                        
    return (starting_nodes, ending_nodes, starting_times)
    
    
def dijkstra_single(graph: Graph, beginning: str, destination: str, time: timedelta) -> None:
    print("\n\n- - - - - - - - - -DIJKSTRA- - - - - - - - - -\n\n")
    print(f"Starting point: {beginning}")
    print(f"Destination: {destination}")
    print(f"Starting time: {time}")

    dijkstra = Algorithms(graph, time, beginning, destination)
    start_time = datetime.now()
    dijkstra.execute_dijkstra()
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    print(f"Dijkstras runtime in seconds: {runtime:.2f}")
    dijkstra.write_solution_to_file('DIJKSTRA', runtime, 'dijkstra_outputs.csv')
    dijkstra.write_runtime_to_file('DIJKSTRA', runtime, 'dijkstra_runtimes.csv')


def dijkstra_sample100(graph: Graph, starting_nodes: list[str], ending_nodes: list[str], starting_times: list[timedelta]) -> None:
    dijkstra_runtimes = []
    print("\n\n- - - - - - - - - -DIJKSTRA- - - - - - - - - -\n\n")
    for i in range(len(starting_nodes)):
        beginning = starting_nodes[i]
        destination = ending_nodes[i]
        time = starting_times[i]
        tmp_graph = graph

        print(f"Starting point: {beginning}")
        print(f"Destination: {destination}")
        print(f"Starting time: {time}")

        dijkstra = Algorithms(tmp_graph, time, beginning, destination)
        start_time = datetime.now()
        dijkstra.execute_dijkstra()
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        print(f"Dijkstras runtime in seconds: {runtime:.2f}")
        dijkstra_runtimes.append(runtime)
        print("")
        dijkstra.write_solution_to_file('DIJKSTRA', runtime, 'dijkstra_outputs.csv')  
        dijkstra.write_runtime_to_file('DIJKSTRA', runtime, 'dijkstra_runtimes.csv') 

    dijkstra_runtimes = trim_extreme_cases(dijkstra_runtimes)
    print(f"Average runtime for finding a path with the dijkstra algorithm, from a random start, to a random destination, on a random time: {np.mean(dijkstra_runtimes)}") 

    
    
def astar_time_single(graph: Graph, beginning: str, destination: str, time: timedelta) -> None:
    print("\n\n- - - - - - - - - -A* : TIME- - - - - - - - - -\n\n")
    print(f"Starting point: {beginning}")
    print(f"Destination: {destination}")
    print(f"Starting time: {time}")

    astar_time = Algorithms(graph, time, beginning, destination)
    start_time = datetime.now()
    astar_time.execute_astar('time')
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    print(f"A* (time criterion) runtime in seconds: {runtime:.2f}")
    astar_time.write_solution_to_file('A* - TIME', runtime, 'astar_time_outputs.csv')
    astar_time.write_runtime_to_file('A* - TIME', runtime, 'astar_time_runtimes.csv')
    
    
def astar_time_sample100(graph: Graph, starting_nodes: list[str], ending_nodes: list[str], starting_times: list[timedelta]) -> None:
    astar_t_runtimes = []
    print("\n\n- - - - - - - - - -A* : TIME- - - - - - - - - -\n\n")
    for i in range(len(starting_nodes)):
        beginning = starting_nodes[i]
        destination = ending_nodes[i]
        time = starting_times[i]
        tmp_graph = graph

        print(f"Starting point: {beginning}")
        print(f"Destination: {destination}")
        print(f"Starting time: {time}")

        astar_time = Algorithms(tmp_graph, time, beginning, destination)
        start_time = datetime.now()
        astar_time.execute_astar('time')
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        print(f"A* (time criterion) runtime in seconds: {runtime:.2f}")
        astar_t_runtimes.append(runtime)
        print("")
        astar_time.write_solution_to_file('A* - TIME', runtime, 'astar_time_outputs.csv')
        astar_time.write_runtime_to_file('A* - TIME', runtime, 'astar_time_runtimes.csv')

    astar_t_runtimes = trim_extreme_cases(astar_t_runtimes)
    print(f"Average runtime for finding a path with the A* algorithm with a time criterion, from a random start, to a random destination, on a random time: {np.mean(astar_t_runtimes)}")

    
def astar_transfers_single(graph: Graph, beginning: str, destination: str, time: timedelta) -> None:
    print("\n\n- - - - - - - - - -A* : TRANSFERS- - - - - - - - - -\n\n")
    print(f"Starting point: {beginning}")
    print(f"Destination: {destination}")
    print(f"Starting time: {time}")

    astar_transfers = Algorithms(graph, time, beginning, destination)
    start_time = datetime.now()
    astar_transfers.execute_astar('transfers')
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    print(f"A* (transfer criterion) runtime in seconds: {runtime:.2f}")
    astar_transfers.write_solution_to_file('A* - TRANSFERS', runtime, 'astar_transfers_outputs.csv')
    astar_transfers.write_runtime_to_file('A* - TRANSFERS', runtime, 'astar_transfers_runtimes.csv')
    
    
def astar_transfers_sample100(graph: Graph, starting_nodes: list[str], ending_nodes: list[str], starting_times: list[timedelta]) -> None:
    astar_t_runtimes = []
    print("\n\n- - - - - - - - - -A* : TRANSFERS- - - - - - - - - -\n\n")
    for i in range(len(starting_nodes)):
        beginning = starting_nodes[i]
        destination = ending_nodes[i]
        time = starting_times[i]
        tmp_graph = graph

        print(f"Starting point: {beginning}")
        print(f"Destination: {destination}")
        print(f"Starting time: {time}")

        astar_transfers = Algorithms(tmp_graph, time, beginning, destination)
        start_time = datetime.now()
        astar_transfers.execute_astar('transfers')
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        print(f"A* (transfer criterion) runtime in seconds: {runtime:.2f}")
        astar_t_runtimes.append(runtime)
        print("")
        astar_transfers.write_solution_to_file('A* - TRANSFERS', runtime, 'astar_transfer_outputs.csv')
        astar_transfers.write_runtime_to_file('A* - TRANSFERS', runtime, 'astar_transfer_runtimes.csv')
        
    astar_t_runtimes = trim_extreme_cases(astar_t_runtimes)
    print(f"Average runtime for finding a path with the A* algorithm with a time criterion, from a random start, to a random destination, on a random time: {np.mean(astar_t_runtimes)}")


if __name__ == '__main__':
    main()

