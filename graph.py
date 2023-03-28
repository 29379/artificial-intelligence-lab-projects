from datetime import datetime, timedelta

class Node:
    def __init__(self, stop_name: str, latitude: float,
                 longitude: float, outgoing_edges: list['Edge']) -> None:
        self.stop_name = stop_name
        self.latitude = latitude
        self.longitude = longitude
        self.outgoing_edges = outgoing_edges
    
    def __str__(self) -> str:
        return "Stop name: " + self.stop_name + ", coordinates: " + \
            self.latitude + ", " + self.longitude
            
    def __lt__(self, other: 'Node') -> bool:
        return self.stop_name < other.stop_name
        
        
class Edge:
    def __init__(self, index: int, company: str, start_node: Node,
                 end_node: Node, line_name: str, 
                 departure_time: timedelta, arrival_time: timedelta) -> None:
        self.index = index
        self.company = company
        self.start_node = start_node
        self.end_node = end_node
        self.line_name = line_name
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        
    def calculate_weight(self, ref_time: datetime.time) -> float:
        dep_time = datetime.combine(datetime.today(), self.departure_time)
        arr_time = datetime.combine(datetime.today(), self.arrival_time)
        if dep_time < datetime.combine(datetime.today(), ref_time):
            return float('inf')
        return (dep_time - datetime.combine(datetime.today(), ref_time)).total_seconds() / 60
    
    def __str__(self) -> str:
        return "Line: " + self.line + " | Departure from " \
            + self.start_node.stop_name + ": " + str(self.departure_time)\
                + " - Arrival at " + self.end_node.stop_name + \
                    ": " + str(self.arrival_time)


class Graph:
    def __init__(self, nodes: dict[str, Node], edges: dict[str, list[Edge]] = None) -> None:
        self.edges = edges or {}
        self.nodes = nodes or {}
        
    def compare_times(self, current: datetime.time, departure: datetime.time) -> bool:
        return current <= departure
            
    def get_outgoing_edges(self, start_node: Node) -> list[Edge]:
        return self.edges.get(start_node.stop_name, [])
    
    def get_neighbor_nodes(self, node:Node) -> list[Node]:
        adjacent_edges = self.get_outgoing_edges(node)
        neighbor_nodes = []
        for edge in adjacent_edges:
            neighbor_nodes.append(edge.end_node)
        return neighbor_nodes
    
    def add_edge(self, edge: Edge) -> None:
        start_node_found = self.nodes[edge.start_node.stop_name]
        end_node_found = self.nodes[edge.end_node.stop_name]
        
        new_edge = Edge(
            index=edge.index,
            company=edge.company,
            start_node=start_node_found,
            end_node=end_node_found,
            line_name=edge.line_name,
            departure_time=edge.departure_time,
            arrival_time=edge.arrival_time
        )
        self.edges[new_edge.start_node.stop_name].append(new_edge)
        if start_node_found:
            start_node_found.outgoing_edges.append(new_edge)  
    
    def useful_connections(self, start_node: Node, time: datetime.time) -> list[Edge]:
        neighbor_edges = self.get_outgoing_edges(start_node)
        output = []
        for edge in neighbor_edges:
            if self.compare_times(time, edge.departure_time):
                output.append(edge)
        return neighbor_edges
            
    def cost_heuristic(self, edge: Edge, time: datetime.time) -> float:
        arrival = datetime.combine(datetime.today(), edge.arrival_time)
        departure = datetime.combine(datetime.today(), edge.departure_time)
        current = datetime.combine(datetime.today(), time)
        
        if self.compare_times(current, departure):
            return float((arrival - departure).total_seconds() / 60) + float((departure - current).total_seconds() / 60)
        return float('inf')
    
    def manhattan_heuristic(self, start: Node, end: Node) -> float:
        return float(abs(start.latitude - end.latitude) + abs(start.longitude - end.longitude))
    
    def euclidean_heuristic(self, start: Node, end: Node) -> float:
        return (float)(pow(start.latitude - end.latitude, 2) + pow(start.longitude - end.longitude, 2))
    
    def cost_of_line_transfer(self, prev: Edge, next: Edge) -> float:
        if prev is not None and next is not None and prev.line_name != next.line_name:
            return 1000
        return 0
           
    def __str__(self) -> str:
        output = ''
        for node in self.nodes.values():
            if node.outgoing_edges.count() > 1:
                output += f"Edges coming from: {str(node)} "
                for edge in node.outgoing_edges:
                    output += f" - {str(edge)}"
        return output


