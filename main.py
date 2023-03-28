from DataReader import DataReader
from graph import *
from Dijkstra import DijkstraAlg
from datetime import datetime
import time


def main():
    graph = DataReader.load_data()
    
    beginning = "PL. GRUNWALDZKI"
    destination = "GALERIA DOMINIKAŃSKA"
    time = datetime(2023, 1, 1, 10, 0, 0).time()
    
    print(len(graph.nodes))
    print(len(graph.edges))
    print(f"Starting point: {beginning}")
    print(f"Destination: {destination}")
    print(f"Starting time: {time}")
    
    # #DijkstraAlg.dijkstra(dij, "Perzowa", "Jaworowa", datetime.strptime("12:30:00", '%H:%M:%S').time())
    # #DijkstraAlg.dijkstra(dij, "Broniewskiego", "BISKUPIN", datetime.strptime("15:00:00", '%H:%M:%S').time())
    # DijkstraAlg.dijkstra(dij, "Perzowa", "Jaworowa", datetime.strptime("23:34:00", '%H:%M:%S').time())

    run = DijkstraAlg(graph, time, beginning, destination)

    #start_time = time.time()    
    shortest_path = run.run_dijkstra()
    #end_time = time.time()
    print(run.print())
    #total_time = end_time - start_time
    #print(f"Total execution time - {str(total_time)} seconds")


if __name__ == '__main__':
    main()

