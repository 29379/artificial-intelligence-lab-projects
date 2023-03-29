from DataReader import DataReader
from graph import *
from Algorithms import Algorithms
from datetime import datetime, timedelta
import random
import numpy as np


def get_samples(graph: Graph) -> tuple[list[str], list[str], list[timedelta]]:
    starting_nodes = random.sample(list(graph.nodes.keys()), 100)
    ending_nodes = random.sample(list(graph.nodes.keys()), 100)
    starting_times = []
    
    for i in range(len(starting_nodes)):
        hours = random.randint(0, 23)
        minutes = random.randint(1, 59)
        starting_times.append(timedelta(hours=hours, minutes=minutes))
        while starting_nodes[i] == ending_nodes[i]:
            ending_nodes[i] = random.choice(list(graph.nodes.keys()))
    return (starting_nodes, ending_nodes, starting_times)


def main() -> None:
    graph = DataReader.load_data()
    starting_nodes, ending_nodes, starting_times = get_samples(graph)
    #   dijkstra_sample100(graph, starting_nodes, ending_nodes, starting_times)
    dijkstra_single(graph, "KRZYKI", "FAT", timedelta(hours=14, minutes=33))
    
    
def dijkstra_single(graph: Graph, beginning: str, destination: str, time: timedelta) -> None:
    print("\n\n- - - - - - - - -DIJKSTRA- - - - - - - - - \n\n")
    print(f"Starting point: {beginning}")
    print(f"Destination: {destination}")
    print(f"Starting time: {time}")

    dijkstra = Algorithms(graph, time, beginning, destination)
    start_time = datetime.now()
    dijkstra.execute_dijkstra()
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    print(f"Dijkstras runtime in seconds: {runtime:.2f}")
    dijkstra.print_solution()


def dijkstra_sample100(graph: Graph, starting_nodes: list[str], ending_nodes: list[str], starting_times: list[timedelta]) -> None:
    dijkstra_runtimes = []
    print("\n\n- - - - - - - - -DIJKSTRA- - - - - - - - - \n\n")
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
        dijkstra.print_solution()
        print("")
    avg = calculate_average_runtimes(dijkstra_runtimes)
    print(f"Average runtime for finding a path with the dijkstra algorithm, from a random start, to a random destination, on a random time: {avg}")


def calculate_average_runtimes(runtimes: list[float]) -> float:
    runtimes.sort()
    n_trim = int(0.05 * len(runtimes))
    start_index = n_trim
    end_index = len(runtimes) - n_trim

    trimmed_runtimes = runtimes[start_index:end_index]
    return np.mean(trimmed_runtimes)


if __name__ == '__main__':
    main()

