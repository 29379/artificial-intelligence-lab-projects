from graph import *
import numpy as np
import datetime
import heapq, sys
from typing import Optional


class DijkstraAlg:
    def __init__(self, graph: Graph, time: datetime.timedelta, start_node: str, end_node: str) -> None:
        self.graph: Graph = graph
        self.time: datetime.timedelta = time
        self.start_node: Node = self.graph.get_node(start_node)
        self.end_node: Node = self.graph.get_node(end_node)
        
        self.cost: dict[str, float] = {}
        self.previous_nodes: dict[str, Optional[Edge]] = {}
        self.lines: dict[str, float] = {}
        self.arrivals: dict[str, datetime.timedelta] = {}
        self.departures: dict[str, datetime.timedelta] = {}
        
        self.settled_nodes: list[Node] = []
        self.unsettled_nodes: list[tuple[float, Node, datetime.timedelta]] = [()]
        
        #   initial setup
        for node_name in self.graph.nodes:
            self.cost[node_name] = float('inf')
            self.previous_nodes[node_name] = None
            self.lines[node_name] = 0
            self.arrivals[node_name] = datetime.timedelta(hours=23, minutes=59, seconds=59)
            self.departures[node_name] = datetime.timedelta(hours=23, minutes=59, seconds=59)

        #   cost at the start == 0, arrival at the start equals time passed into dijkstra
        self.cost[self.start_node.stop_name] = 0
        self.arrivals[self.start_node.stop_name] = time
        
    def execute(self) -> None:
        while self.unsettled_nodes:
            priority, my_focus, current_time = heapq.heappop(self.unsettled_nodes)
            self.settled_nodes.append(my_focus)
            if my_focus == self.end_node:
                break
            else:
                self.update_unsettled(my_focus, current_time) 
        self.retrieve_solution()

    def update_unsettled(self, my_focus: Node, current_time: datetime.timedelta) -> None:
        for edge_list in self.graph.edges[my_focus.stop_name]:
            for edge in edge_list:
                #   checking if the edge is valid given current time
                if (self.lines[my_focus.stop_name] == 0 and current_time == edge.departure_time) \
                    or (edge.line_name != self.lines[my_focus.stop_name] and current_time < edge.departure_time) \
                    or (edge.line_name == self.lines[my_focus.stop_name] and current_time == edge.departure_time):
                        new_cost = (edge.arrival_time - self.time).seconds / 60
                        end_name = edge.end_node.stop_name
                        #   checking if I was there already and if not, is it worth it to go there
                        if edge.end_node not in self.settled_nodes and new_cost < self.cost[end_name]:
                            self.arrivals[end_name] = edge.arrival_time
                            self.departures[end_name] = edge.departure_time
                            self.lines[end_name] = edge.line_name
                            self.cost[end_name] = new_cost  #   new priority
                            self.previous_nodes[end_name] = my_focus
                            heapq.heappush(self.unsettled_nodes, (new_cost, edge.end_node, edge.arrival_time))
                            
                            
                
    
    def retrieve_solution(self) -> None:
        pass
    
    def print_solution(self) -> None:
        pass
        
        
    # def print(self) -> str:
    #     result = []
    #     for node in self.graph.nodes:
    #         result.append("Node {}:\n".format(node))
    #         for edges in self.graph.nodes.values():
    #             for edge in edges.outgoing_edges:
    #                 result.append(f"\t---> Node {edge.end_node.stop_name} (distance: {edge.calculate_weight(self.time)})\n")
    #     return "".join(result)

    # def find_shortest_path(self, my_focus: Node) -> None:
    #     for edge in my_focus.outgoing_edges:
    #         neighbor = edge.end_node
            
    #         #   if the neighbor was settled already - skip the following logic
    #         if not self.settled_nodes.contains(neighbor):
    #             new_distance = self.shortest_paths[my_focus][0] + edge.calculate_weight(self.time)
    #             #   if neighbor was not settled yet, or if new_distance < current_distance, update neighbors distance and path
    #             if neighbor not in self.shortest_paths or new_distance < self.shortest_paths[neighbor][0]:
    #                 self.shortest_paths[neighbor] = (new_distance, self.shortest_paths[my_focus][1] + [edge])
    #                 #   updating the priorities in the queue
    #                 self.unsettled_nodes.enqueue(neighbor, self.shortest_paths[neighbor][0])
                    
    # def run_dijkstra(self) -> tuple[int, list[Edge]]:
    #     self.shortest_paths[self.start_node] = (0, [])
    #     for node in self.graph.nodes.values():
    #         if not node == self.start_node:
    #             self.unsettled_nodes.enqueue(node, float('inf'))
    #         else:
    #             self.unsettled_nodes.enqueue(node, 0)
    #     while not self.unsettled_nodes.is_empty():
    #         #   get a Node with the lowest priority
    #         my_focus = self.unsettled_nodes.dequeue()
    #         self.settled_nodes.enqueue(my_focus, 0)
    #         self.find_shortest_path(my_focus)
            
    #         if my_focus == self.end_node:
    #             return self.shortest_paths.get(self.end_node)
    #     #   if after searching through the whole graph we still didnt find anything
    #     return None
    
    # def print_solution(self) -> None:
    #     if self.end_node in self.shortest_paths and self.shortest_paths[self.end_node] == (0, []):
    #         print("No solution was found")
    #     else:
    #         distance, edges = self.shortest_paths.get(self.end_node, (float('inf'), []))
    #         if len(edges) == 0:
    #             print("Something went wrong, no buses were used")
    #         else:
    #             print(f"Shortest path from {str(self.start_node)} to {str(self.end_node)}")
    #             for edge in edges:
    #                 print(f"\t- {edge.start_node.stop_name} -> {edge.end_node.stop_name}: {edge.calculate_weight(self.time)}")
    #             print(f"Total distance: {distance} minutes")
    #             print(f">   Departure time: {str(self.time)}")
    #             print(f">   Arrival time: {str(edges[-1].arrival_time)}")
                        
        