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
    
    #   dijkstra_sample100(graph, starting_nodes, ending_nodes, starting_times)
    #   astar_time_sample100(graph, starting_nodes, ending_nodes, starting_times)
    astar_transfers_sample100(graph, starting_nodes, ending_nodes, starting_times)
    #   dijkstra_single(graph, "Psie Pole", "FAT", timedelta(hours=14, minutes=33))
    #   astar_time_single(graph, "Psie Pole", "FAT", timedelta(hours=14, minutes=33))
    #   astar_transfers_single(graph, "Psie Pole", "FAT", timedelta(hours=14, minutes=33))


def calculate_average_runtimes(runtimes: list[float]) -> float:
    runtimes.sort()
    n_trim = int(0.05 * len(runtimes))
    start_index = n_trim
    end_index = len(runtimes) - n_trim

    trimmed_runtimes = runtimes[start_index:end_index]
    return np.mean(trimmed_runtimes)


def get_samples(graph: Graph) -> tuple[list[str], list[str], list[timedelta]]:
    starting_nodes = random.sample(list(graph.nodes.keys()), 100)
    ending_nodes = random.sample(list(graph.nodes.keys()), 100)
    starting_times = []
    
    for i in range(len(starting_nodes)):
        hours = random.randint(0, 23)
        minutes = random.randint(1, 59)
        starting_times.append(timedelta(hours=hours, minutes=minutes))
        while starting_nodes[i] == ending_nodes[i] \
            or starting_nodes[i] == "Zorawina - Niepodleglosci (Mostek)" \
            or ending_nodes[i] == "Zorawina - Niepodleglosci (Mostek)":
                starting_nodes[i] = random.choice(list(graph.nodes.keys()))
                ending_nodes[i] = random.choice(list(graph.nodes.keys()))
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
    dijkstra.write_solution_to_file('DIJKSTRA', runtime)


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
        dijkstra.write_solution_to_file('DIJKSTRA', runtime)   
    avg = calculate_average_runtimes(dijkstra_runtimes)
    print(f"Average runtime for finding a path with the dijkstra algorithm, from a random start, to a random destination, on a random time: {avg}") 
    
    dijkstra_runtimes.sort()
    n_trim = int(0.05 * len(dijkstra_runtimes))
    start_index = n_trim
    end_index = len(dijkstra_runtimes) - n_trim
    trimmed_runtimes = dijkstra_runtimes[start_index:end_index]
    
    with open('runtimes.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for elem in trimmed_runtimes:
                writer.writerow(['DIJKSTRA', elem])
            writer.writerow('')

    
    
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
    astar_time.write_solution_to_file('A* - TIME', runtime)
    
    
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
        astar_time.write_solution_to_file('A* - TIME', runtime)
    avg = calculate_average_runtimes(astar_t_runtimes)
    print(f"Average runtime for finding a path with the A* algorithm with a time criterion, from a random start, to a random destination, on a random time: {avg}")
    
    astar_t_runtimes.sort()
    n_trim = int(0.05 * len(astar_t_runtimes))
    start_index = n_trim
    end_index = len(astar_t_runtimes) - n_trim
    trimmed_runtimes = astar_t_runtimes[start_index:end_index]
    #   random.shuffle(trimmed_runtimes)
    
    with open('runtimes.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for elem in trimmed_runtimes:
                writer.writerow(['A* - TIME', elem])
            writer.writerow('')

    
    
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
    astar_transfers.write_solution_to_file('A* - TRANSFERS', runtime)
    
    
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
        astar_transfers.write_solution_to_file('A* - TRANSFERS', runtime)
    avg = calculate_average_runtimes(astar_t_runtimes)
    print(f"Average runtime for finding a path with the A* algorithm with a time criterion, from a random start, to a random destination, on a random time: {avg}")
    
    astar_t_runtimes.sort()
    n_trim = int(0.05 * len(astar_t_runtimes))
    start_index = n_trim
    end_index = len(astar_t_runtimes) - n_trim
    trimmed_runtimes = astar_t_runtimes[start_index:end_index]
    random.sh
    
    with open('runtimes.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for elem in trimmed_runtimes:
                writer.writerow(['A* - TRANSFERS', elem])
            writer.writerow('')
    


if __name__ == '__main__':
    main()

