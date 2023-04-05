from DataReader import DataReader
from graph import *
from Algorithms import Algorithms
from datetime import datetime, timedelta
import random, csv, math
import numpy as np


def main() -> None:
    graph = DataReader.load_data()
    starting_nodes, ending_nodes, starting_times = get_samples(graph)
    
    with open('dijkstra_outputs.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Stop', 'Runtime', 'Line', 'Departure time', 'Arrival time'])
        
    with open('astar_time_outputs.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Stop', 'Runtime', 'Line', 'Departure time', 'Arrival time'])
        
    with open('astar_transfer_outputs.csv', 'a', newline='') as file:
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
    # astar_transfers_single(graph, "Warminska", "Kielczow - WODROL", timedelta(hours=21, minutes=19))
    

def trim_extreme_cases(runtimes: list[float]) -> list[int]:
    n = len(runtimes)
    n_trim = min(5, math.ceil(0.05 * n))

    top_indices = []
    bottom_indices = []
    
    for i in range(n_trim):
        top_runtime = float('-inf')
        bottom_runtime = float('inf')
        top_index = None
        bottom_index = None
        for j, runtime in enumerate(runtimes):
            if runtime is None:
                continue
            if runtime > top_runtime and j not in top_indices:
                top_runtime = runtime
                top_index = j
            if runtime < bottom_runtime and j not in bottom_indices:
                bottom_runtime = runtime
                bottom_index = j
        if top_index is not None:
            top_indices.append(top_index)
            runtimes[top_index] = None
        if bottom_index is not None:
            bottom_indices.append(bottom_index)
            runtimes[bottom_index] = None
    
    return top_indices + bottom_indices


def trim_extreme_cases_in_file(file_name: str) -> None:
    with open(file_name, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    runtimes = [float(row[1]) for row in data[1:]]
    indicies_to_trim = trim_extreme_cases(runtimes)

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for i in range(len(data)):
            if i not in indicies_to_trim:
                writer.writerow(data[i])
        #writer.writerows(data)


def get_samples(graph: Graph) -> tuple[list[str], list[str], list[timedelta]]:
    starting_nodes = random.sample(list(graph.nodes.keys()), 100)
    ending_nodes = random.sample(list(graph.nodes.keys()), 100)
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

    dijkstra = Algorithms(graph, time, beginning, destination, [])
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

        dijkstra = Algorithms(tmp_graph, time, beginning, destination, [])
        start_time = datetime.now()
        dijkstra.execute_dijkstra()
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        print(f"Dijkstras runtime in seconds: {runtime:.2f}")
        dijkstra_runtimes.append(runtime)
        print("")
        dijkstra.write_solution_to_file('DIJKSTRA', runtime, 'dijkstra_outputs.csv')  
        dijkstra.write_runtime_to_file('DIJKSTRA', runtime, 'dijkstra_runtimes.csv') 

    trim_extreme_cases_in_file('dijkstra_runtimes.csv')
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

        astar_time = Algorithms(tmp_graph, time, beginning, destination, [])
        start_time = datetime.now()
        astar_time.execute_astar('time')
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        print(f"A* (time criterion) runtime in seconds: {runtime:.2f}")
        astar_t_runtimes.append(runtime)
        print("")
        astar_time.write_solution_to_file('A* - TIME', runtime, 'astar_time_outputs.csv')
        astar_time.write_runtime_to_file('A* - TIME', runtime, 'astar_time_runtimes.csv')

    trim_extreme_cases_in_file('astar_time_runtimes.csv')
    print(f"Average runtime for finding a path with the A* algorithm with a time criterion, from a random start, to a random destination, on a random time: {np.mean(astar_t_runtimes)}")

    
def astar_transfers_single(graph: Graph, beginning: str, destination: str, time: timedelta) -> None:
    print("\n\n- - - - - - - - - -A* : TRANSFERS- - - - - - - - - -\n\n")
    print(f"Starting point: {beginning}")
    print(f"Destination: {destination}")
    print(f"Starting time: {time}")

    astar_transfers = Algorithms(graph, time, beginning, destination, [])
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

        astar_transfers = Algorithms(tmp_graph, time, beginning, destination, [])
        start_time = datetime.now()
        astar_transfers.execute_astar('transfers')
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        print(f"A* (transfer criterion) runtime in seconds: {runtime:.2f}")
        astar_t_runtimes.append(runtime)
        print("")
        astar_transfers.write_solution_to_file('A* - TRANSFERS', runtime, 'astar_transfer_outputs.csv')
        astar_transfers.write_runtime_to_file('A* - TRANSFERS', runtime, 'astar_transfer_runtimes.csv')
        
    trim_extreme_cases_in_file('astar_transfer_runtimes.csv')
    print(f"Average runtime for finding a path with the A* algorithm with a time criterion, from a random start, to a random destination, on a random time: {np.mean(astar_t_runtimes)}")


if __name__ == '__main__':
    main()

