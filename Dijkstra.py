from graph import *
import numpy as np
from datetime import datetime
import heapq, sys
from typing import Optional


class DijkstraAlg:
    def __init__(self, graph: Graph, time: datetime.time, start_node: str, end_node: str) -> None:
        self.graph: Graph = graph
        self.time: datetime.time = time
        self.settled_nodes: PriorityQueue = PriorityQueue()
        self.unsettled_nodes: PriorityQueue = PriorityQueue()
        self.distances: dict[Node, float] = {}
        self.previous: dict[Node, Optional[Edge]] = {}
        self.start_node: Node = self.graph.get_node(start_node)
        self.end_node: Node = self.graph.get_node(end_node)
        self.shortest_paths: dict[Node, tuple[float, list[Edge]]] = {}
        
        for node in self.graph.nodes.values():
            self.distances[node] = float('inf')
            self.previous[node] = None
        self.distances[self.graph.get_node(start_node)] = 0
        
    def print(self) -> str:
        result = []
        for node in self.graph.nodes:
            result.append("Node {}:\n".format(node))
            for edges in self.graph.nodes.values():
                for edge in edges.outgoing_edges:
                    result.append(f"\t---> Node {edge.end_node.stop_name} (distance: {edge.calculate_weight(self.time)})\n")
        return "".join(result)

    def find_shortest_path(self, my_focus: Node) -> None:
        for edge in my_focus.outgoing_edges:
            neighbor = edge.end_node
            
            #   if the neighbor was settled already - skip the following logic
            if not self.settled_nodes.contains(neighbor):
                new_distance = self.shortest_paths[my_focus][0] + edge.calculate_weight(self.time)
                #   if neighbor was not settled yet, or if new_distance < current_distance, update neighbors distance and path
                if neighbor not in self.shortest_paths or new_distance < self.shortest_paths[neighbor][0]:
                    self.shortest_paths[neighbor] = (new_distance, self.shortest_paths[my_focus][1] + [edge])
                    #   updating the priorities in the queue
                    self.unsettled_nodes.enqueue(neighbor, self.shortest_paths[neighbor][0])
                    
    def run_dijkstra(self) -> tuple[int, list[Edge]]:
        self.shortest_paths[self.start_node] = (0, [])
        for node in self.graph.nodes.values():
            if not node == self.start_node:
                self.unsettled_nodes.enqueue(node, float('inf'))
            else:
                self.unsettled_nodes.enqueue(node, 0)
        while not self.unsettled_nodes.is_empty():
            #   get a Node with the lowest priority
            my_focus = self.unsettled_nodes.dequeue()
            self.settled_nodes.enqueue(my_focus, 0)
            self.find_shortest_path(my_focus)
            
            if my_focus == self.end_node:
                return self.shortest_paths.get(self.end_node)
        #   if after searching through the whole graph we still didnt find anything
        return None
    
    def print_solution(self) -> None:
        if self.end_node in self.shortest_paths and self.shortest_paths[self.end_node] == (0, []):
            print("No solution was found")
        else:
            distance, edges = self.shortest_paths.get(self.end_node, (float('inf'), []))
            if len(edges) == 0:
                print("Something went wrong, no buses were used")
            else:
                print(f"Shortest path from {str(self.start_node)} to {str(self.end_node)}")
                for edge in edges:
                    print(f"\t- {edge.start_node.stop_name} -> {edge.end_node.stop_name}: {edge.calculate_weight(self.time)}")
                print(f"Total distance: {distance} minutes")
                print(f">   Departure time: {str(self.time)}")
                print(f">   Arrival time: {str(edges[-1].arrival_time)}")
                        
        