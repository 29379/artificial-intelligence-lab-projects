from graph import *
import numpy as np
import datetime
import heapq, sys, math
from typing import Optional

class Algorithms:
    def __init__(self, graph: Graph, time: datetime.timedelta, start_node: str, end_node: str) -> None:
        self.graph: Graph = graph
        self.time: datetime.timedelta = time
        self.start_node: Node = self.graph.get_node(start_node)
        self.end_node: Node = self.graph.get_node(end_node)
        
        self.cost: dict[str, float] = {}
        self.previous_nodes: dict[str, Optional[str]] = {}
        self.lines: dict[str, any] = {}
        self.arrivals: dict[str, datetime.timedelta] = {}
        self.departures: dict[str, datetime.timedelta] = {}
        
        self.settled_nodes: list[str] = []
        self.unsettled_nodes: list[tuple[float, str, datetime.timedelta]] = []
        
        self.path: list[any] = []
        self.arr_times: list[datetime.datetime] = []
        self.dep_times: list[datetime.datetime] = []
        self.used_lines: list[any] = []
        
        #   initial setup
        for node_name in self.graph.nodes:
            self.cost[node_name] = float('inf')
            self.previous_nodes[node_name] = None
            self.lines[node_name] = 0
            self.arrivals[node_name] = datetime.timedelta(hours=23, minutes=59, seconds=59)
            self.departures[node_name] = datetime.timedelta(hours=23, minutes=59, seconds=59)

        self.unsettled_nodes.append((0, self.start_node.stop_name, self.time))
        #   cost at the start == 0, arrival at the start equals time passed into dijkstra
        self.cost[self.start_node.stop_name] = 0
        self.arrivals[self.start_node.stop_name] = time
        
        
    def retrieve_solution(self, my_focus: str) -> None:
        while my_focus is not None:
            self.arr_times.append(datetime.datetime.min + self.arrivals[my_focus])
            self.dep_times.append(datetime.datetime.min + self.departures[my_focus])
            self.used_lines.append(self.lines[my_focus])
            self.path.append(my_focus)
            my_focus = self.previous_nodes[my_focus]
        self.arr_times.reverse()
        self.dep_times.reverse()
        self.used_lines.reverse()
        self.path.reverse()            
    
    
    def print_solution(self) -> None:
        print(self.path[0])
        for i in range(1, len(self.path)):
            print(f">   {self.used_lines[i]} :   {self.dep_times[i].strftime('%H:%M:%S')}  -   {self.arr_times[i].strftime('%H:%M:%S')}")
            print(self.path[i])
            
              
    def execute_dijkstra(self) -> None:
        while self.unsettled_nodes:
            priority, my_focus, current_time = heapq.heappop(self.unsettled_nodes)
            self.settled_nodes.append(my_focus)
            if my_focus == self.end_node.stop_name:
                break
            else:
                if my_focus == "Zorawina - Niepodleglosci (Mostek)":
                    continue
                else:
                    self.update_unsettled_dijkstra(my_focus, current_time) 
        self.retrieve_solution(my_focus)


    def update_unsettled_dijkstra(self, my_focus: str, current_time: datetime.timedelta) -> None:
        for edge in self.graph.edges[my_focus]:
            total_seconds = edge.departure_time.total_seconds()
            cmp_tool = (datetime.datetime.min + datetime.timedelta(seconds=total_seconds) - datetime.datetime.min)
            
            #   checking if the edge is valid given current time
            if (self.lines[my_focus] == 0 and current_time == cmp_tool) \
                or (edge.line_name != self.lines[my_focus] and current_time < cmp_tool) \
                or (edge.line_name == self.lines[my_focus] and current_time == cmp_tool):
                    new_cost = (edge.arrival_time - self.time).seconds / 60
                    end_name = edge.end_node.stop_name
                    #   checking if I was there already and if not, is it worth it to go there
                    if edge.end_node not in self.settled_nodes and new_cost < self.cost[end_name]:
                        self.arrivals[end_name] = edge.arrival_time
                        self.departures[end_name] = edge.departure_time
                        self.lines[end_name] = edge.line_name
                        self.previous_nodes[end_name] = my_focus
                        self.cost[end_name] = new_cost  #   new priority
                        heapq.heappush(self.unsettled_nodes, (new_cost, edge.end_node.stop_name, edge.arrival_time))
                        
                        
    def manhattan_dist(self, start: str, end: str) -> float:
        return float(abs(self.graph.nodes[start].latitude - self.graph.nodes[end].latitude) 
                     + abs(self.graph.nodes[start].longitude - self.graph.nodes[end].longitude))


    def euclidean_dist(self, start: str, end: str) -> float:
        return float(math.sqrt(pow(self.graph.nodes[start].latitude - self.graph.nodes[end].latitude, 2) 
                         + pow(self.graph.nodes[start].longitude - self.graph.nodes[end].longitude, 2)))


    def cost_of_line_transfer(self, prev: Edge, next: Edge) -> float:
        if prev is not None and next is not None and prev.line_name != next.line_name:
            return 1000
        return 0
    
    
    #   'criterion' string decides which crierion will be used in the a* algorithm,
    #   which helps me reduce excess code, since the initial setup is the same
    def execute_astar(self, criterion: str) -> None:
        while self.unsettled_nodes:
            priority, my_focus, current_time = heapq.heappop(self.unsettled_nodes)
            self.settled_nodes.append(my_focus)
            if my_focus == self.end_node.stop_name:
                break
            else:
                if my_focus == "Zorawina - Niepodleglosci (Mostek)":
                    continue
                else:
                    if criterion == 'time':
                        self.update_unsettled_astar_time(priority, my_focus, current_time) 
                    elif criterion == 'transfers':
                        self.update_unsettled_astar_transfers(priority, my_focus, current_time) 
                    else:
                        print("Wrong criterion!")
        self.retrieve_solution(my_focus)
        
    
    def update_unsettled_astar_time(self, priority: any, my_focus: str, current_time: datetime.timedelta) -> None:
        for edge in self.graph.edges[my_focus]:
            total_seconds = edge.departure_time.total_seconds()
            cmp_tool = (datetime.datetime.min + datetime.timedelta(seconds=total_seconds) - datetime.datetime.min)
            
            if (self.lines[my_focus] == 0 and current_time == cmp_tool) \
                or (edge.line_name != self.lines[my_focus] and current_time < cmp_tool) \
                or (edge.line_name == self.lines[my_focus] and current_time == cmp_tool):
                    new_cost = (edge.arrival_time - self.time).seconds / 60
                    end_name = edge.end_node.stop_name
                    if edge.end_node not in self.settled_nodes and new_cost < self.cost[end_name]:
                        self.arrivals[end_name] = edge.arrival_time
                        self.departures[end_name] = edge.departure_time
                        self.lines[end_name] = edge.line_name
                        self.previous_nodes[end_name] = my_focus
                        self.cost[end_name] = new_cost  
                        
                        priority = new_cost + self.manhattan_dist(my_focus, end_name)
                        heapq.heappush(self.unsettled_nodes, (priority, edge.end_node.stop_name, edge.arrival_time))
                        
                            
    def update_unsettled_astar_transfers(self, priority: any, my_focus: str, current_time: datetime.timedelta) -> None:
            for edge in self.graph.edges[my_focus]:
                total_seconds = edge.departure_time.total_seconds()
                cmp_tool = (datetime.datetime.min + datetime.timedelta(seconds=total_seconds) - datetime.datetime.min)
                
                if (self.lines[my_focus] == 0 and current_time == cmp_tool) \
                    or (edge.line_name != self.lines[my_focus] and current_time < cmp_tool) \
                    or (edge.line_name == self.lines[my_focus] and current_time == cmp_tool):
                        new_cost = (edge.arrival_time - self.time).seconds / 60
                        if edge.line_name != self.lines[my_focus]:
                            new_cost += 25
                        end_name = edge.end_node.stop_name
                        if edge.end_node not in self.settled_nodes and new_cost < self.cost[end_name]:
                            self.arrivals[end_name] = edge.arrival_time
                            self.departures[end_name] = edge.departure_time
                            self.lines[end_name] = edge.line_name
                            self.previous_nodes[end_name] = my_focus
                            self.cost[end_name] = new_cost  
                            
                            priority = new_cost + self.manhattan_dist(my_focus, end_name)
                            heapq.heappush(self.unsettled_nodes, (priority, edge.end_node.stop_name, edge.arrival_time))
    