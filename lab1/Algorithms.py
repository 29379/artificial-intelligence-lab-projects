from graph import *
import numpy as np
import datetime
import heapq, sys, math, csv, random
from typing import Optional
import pandas as pd

class Algorithms:
    def __init__(self, graph: Graph, time: datetime.timedelta, start_node: str, end_node: str, tabu_stops: list[str]) -> None:
        #   params
        self.graph: Graph = graph
        self.time: datetime.timedelta = time
        self.start_node: Node = self.graph.get_node(start_node)
        self.end_node: Node = self.graph.get_node(end_node)
        
        #   params for tabu search
        self.tabu_stops: list[str] = tabu_stops
        self.tabu_current_solution = tabu_stops
        random.shuffle(tabu_stops)
        self.tabu_best_solution = self.tabu_current_solution
        self.tabu_best_solution_cost = 0
        self.tabu_max_iterations = math.ceil(1.1*(len(self.tabu_stops) * len(self.tabu_stops)))
        self.tabu_improvement_threshold = 2 * math.floor(math.sqrt(self.tabu_max_iterations))
        self.tabu_turns_improved = 0
        self.tabu_tenure = len(self.tabu_stops)
        self.tabu_list = []
        self.tabu_best_path = None
        self.tabu_best_arrival = None
        self.tabu_best_departure = None
        self.tabu_best_line = None
        
        
        self.cost: dict[str, float] = {}    #   weights
        self.previous_nodes: dict[str, Optional[str]] = {}  #   to check which stop was visited before
        self.lines: dict[str, any] = {} #   to check how did I get to wherever I am rn (through which edge)
        self.arrivals: dict[str, datetime.timedelta] = {}   #   to check the time of arrival at a stop
        self.departures: dict[str, datetime.timedelta] = {} #   to check the time of departure from a stop
        
        #   priority queue equivalents
        self.settled_nodes: list[str] = []
        self.unsettled_nodes: list[tuple[float, str, datetime.timedelta]] = []  #   priority, current node, current time
        
        #   variables used to keep track of the best solution at the moment
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
            
            if len(self.path) > 250:
                break
        
        self.arr_times.reverse()
        self.dep_times.reverse()
        self.used_lines.reverse()
        self.path.reverse()            
    
    
    def write_solution_to_file(self, alg_type: str, runtime: str, file_name: str) -> None:
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([alg_type, runtime])
            writer.writerow(['From: ', self.start_node.stop_name, 'To: ', self.end_node.stop_name])
            
            #   if len(self.path) > 250, then the path is nonsensical either way it is 
            #   Wrocław after all, nobody will ride 250 stops, no matter the case
            #   the value is kinda arbitrary, but makes sense
            if len(self.path) > 250:
                writer.writerow(["The correct path was not found",'','','',''])
            else:
                for i in range(0, len(self.path)):
                    if i < len(self.path):
                        if i == 0:
                            writer.writerow([self.path[i], '>', '', '', self.arr_times[i].strftime('%H:%M:%S')])
                        else:
                            writer.writerow([self.path[i], '>', self.used_lines[i], self.dep_times[i].strftime('%H:%M:%S'), self.arr_times[i].strftime('%H:%M:%S')])
                    else:
                        writer.writerow([self.path[i]])
            writer.writerow('')
            
            
    def write_runtime_to_file(self, alg_type: str, runtime: str, file_name: str) -> None:
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            if len(self.path) > 250:
                writer.writerow([alg_type, float('inf'), self.start_node.stop_name, '--->', self.end_node.stop_name, self.time])
            else:
                writer.writerow([alg_type, runtime, self.start_node.stop_name, '--->', self.end_node.stop_name, self.time])
            
              
    def execute_dijkstra(self) -> None:
        while self.unsettled_nodes:
            priority, my_focus, current_time = heapq.heappop(self.unsettled_nodes)

            if not my_focus in self.settled_nodes:
                self.settled_nodes.append(my_focus)
            
            if my_focus == self.end_node.stop_name:
                break
            else:
                self.update_unsettled_dijkstra(my_focus, current_time) 
        self.retrieve_solution(my_focus)


    def update_unsettled_dijkstra(self, my_focus: str, current_time: datetime.timedelta) -> None:
        for edge in self.graph.edges[my_focus]:
            departure_total_seconds = datetime.timedelta(seconds=edge.departure_time.total_seconds())
            
            start_journey_condition = (self.lines[my_focus] == 0) and (current_time <= departure_total_seconds) and ((departure_total_seconds - current_time).total_seconds() <= 7200)
            transfer_condition = (edge.line_name != self.lines[my_focus]) and (current_time < departure_total_seconds) and ((departure_total_seconds - current_time).total_seconds() <= 7200)
            keep_going_condition = (edge.line_name == self.lines[my_focus]) and (current_time == departure_total_seconds)
            
            if start_journey_condition or transfer_condition or keep_going_condition:
                    new_cost = (edge.arrival_time - self.time).seconds
                    end_name = edge.end_node.stop_name
                    #   checking if I was there already and if not, is it worth it to go there
                    if edge.end_node not in self.settled_nodes and new_cost < self.cost[end_name]:
                        self.arrivals[end_name] = edge.arrival_time
                        self.departures[end_name] = edge.departure_time
                        self.lines[end_name] = edge.line_name
                        self.previous_nodes[end_name] = my_focus
                        self.cost[end_name] = new_cost  #   new__cost == new priority
                        heapq.heappush(self.unsettled_nodes, (new_cost, edge.end_node.stop_name, edge.arrival_time))
                        
                        
    def manhattan_dist(self, start: str, end: str) -> float:
        return float(abs(self.graph.nodes[start].latitude - self.graph.nodes[end].latitude) 
                     + abs(self.graph.nodes[start].longitude - self.graph.nodes[end].longitude))


    def euclidean_dist(self, start: str, end: str) -> float:
        return float(math.sqrt(pow(self.graph.nodes[start].latitude - self.graph.nodes[end].latitude, 2) 
                         + pow(self.graph.nodes[start].longitude - self.graph.nodes[end].longitude, 2)))
    
    
                            
    #   'criterion' string decides which crierion will be used in the a* algorithm,
    #   which helps me reduce excess code, since the initial setup is the same
    def execute_astar(self, criterion: str) -> None:
        while self.unsettled_nodes:
            priority, my_focus, current_time = heapq.heappop(self.unsettled_nodes)
            self.settled_nodes.append(my_focus)
            if my_focus == self.end_node.stop_name:
                break
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
            departure_total_seconds = datetime.timedelta(seconds=edge.departure_time.total_seconds())
            
            start_journey_condition = (self.lines[my_focus] == 0) and (current_time <= departure_total_seconds) and ((departure_total_seconds - current_time).total_seconds() <= 7200)
            transfer_condition = (edge.line_name != self.lines[my_focus]) and (current_time < departure_total_seconds) and ((departure_total_seconds - current_time).total_seconds() <= 7200)
            keep_going_condition = (edge.line_name == self.lines[my_focus]) and (current_time == departure_total_seconds)
            
            if start_journey_condition or transfer_condition or keep_going_condition:        
                    
                    end_name = edge.end_node.stop_name
                    new_cost = (edge.arrival_time - self.time).seconds
                                        
                    if edge.end_node not in self.settled_nodes and new_cost < self.cost[end_name]:
                        self.arrivals[end_name] = edge.arrival_time
                        self.departures[end_name] = edge.departure_time
                        self.lines[end_name] = edge.line_name
                        self.previous_nodes[end_name] = my_focus
                        self.cost[end_name] = new_cost  
                        
                        priority = new_cost + (self.manhattan_dist(my_focus, end_name) * 1000)
                        heapq.heappush(self.unsettled_nodes, (priority, edge.end_node.stop_name, edge.arrival_time))
                                                        
                            
    
    def update_unsettled_astar_transfers(self, priority: any, my_focus: str, current_time: datetime.timedelta) -> None:
            for edge in self.graph.edges[my_focus]:
                departure_total_seconds = datetime.timedelta(seconds=edge.departure_time.total_seconds())
                
                start_journey_condition = (self.lines[my_focus] == 0) and (current_time <= departure_total_seconds) and ((departure_total_seconds - current_time).total_seconds() <= 7200)
                transfer_condition = (edge.line_name != self.lines[my_focus]) and (current_time < departure_total_seconds) and ((departure_total_seconds - current_time).total_seconds() <= 7200)
                keep_going_condition = (edge.line_name == self.lines[my_focus]) and (current_time == departure_total_seconds)
                
                if start_journey_condition or transfer_condition or keep_going_condition:
                        
                        end_name = edge.end_node.stop_name
                        new_cost = (edge.arrival_time - self.time).seconds
                        
                        #   1 minnute == 60 in priority
                        if edge.line_name != self.lines[my_focus] and self.lines[my_focus] != 0:
                            new_cost += 600
                        
                        if edge.end_node not in self.settled_nodes and new_cost < self.cost[end_name]:
                            self.arrivals[end_name] = edge.arrival_time
                            self.departures[end_name] = edge.departure_time
                            self.lines[end_name] = edge.line_name
                            self.previous_nodes[end_name] = my_focus
                            self.cost[end_name] = new_cost
                            
                            priority = new_cost + (self.manhattan_dist(my_focus, end_name) * 1000)
                            heapq.heappush(self.unsettled_nodes, (priority, edge.end_node.stop_name, edge.arrival_time))
                            
                            
                            
    # def tabu_search(self, criterion: str) -> None:
    #     self.tabu_best_solution_cost, self.best_path, self.best_arrival, self.best_departure, self.tabu_best_line = self.get_path_cost(criterion)
        
    #     for _ in range(self.tabu_max_iterations):
    #         if self.tabu_turns_improved > self.tabu_improvement_threshold:
    #             break
    #         best_neighbors = None
    #         best_neighbors_cost = float('inf')
    #         best_neighbors_path = []
    #         best_neighbors_arrival = []
    #         best_neighbors_departure = []
    #         best_neighbors_line = []
    #         x_coord, y_coord = 0, 0
            
    #         for i in range(len(self.tabu_stops)):
    #             for j in range(i+1, self.tabu_stops):
    #                 neighbors = self.tabu_current_solution
    #                 temp = neighbors[i]
    #                 neighbors[i] = neighbors[j]
    #                 neighbors[j] = temp
    #                 neighbor_cost, neighbor_path, neighbor_arrival, neighbor_departure, neighbor_line = self.get_path_cost(criterion)
    #                 if (i, j) not in self.tabu_list:
    #                     if neighbor_cost < best_neighbors_cost:
    #                         best_neighbors = neighbors
    #                         best_neighbors_path = neighbor_path
    #                         best_neighbors_arrival = neighbor_arrival
    #                         best_neighbors_departure = neighbor_departure
    #                         best_neighbors_line = neighbor_line
    #                         x_coord = i
    #                         y_coord = j
    #             self.tabu_list.append((x_coord, y_coord))

    #         if best_neighbors is not None:
    #             self.tabu_current_solution = best_neighbors
    #             self.tabu_list.append((x_coord, y_coord))
                
    #         if len(self.tabu_list > self.tabu_tenure):
    #             self.tabu_list.pop(0)
            
    #         if best_neighbors_cost < self.tabu_best_solution_cost:
    #             self.tabu_best_solution = best_neighbors
    #             self.tabu_best_solution_cost = best_neighbors_cost
    #             self.tabu_best_path = best_neighbors_path
    #             self.tabu_best_arrival = best_neighbors_arrival
    #             self.tabu_best_departure = best_neighbors_departure
    #             self.tabu_best_line = best_neighbors_line
    #             self.tabu_turns_improved = 0
    #         else:
    #             self.tabu_turns_improved += 1
                
    #         #   write the solution to file or something
        
        
    # def get_path_cost(self, criterion: str):
    #     curr_time = self.time
    #     curr_stop = self.start_node
    #     final_path = [self.start_node]
        
    #     final_cost = 0
    #     final_arrival_time = ['']
    #     final_departure_time = ['']
    #     final_line = ['']

    #     for stop in self.tabu_stops:
    #         pass

            
