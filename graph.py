from datetime import datetime, timedelta

class Node:
    def __init__(self, stop_name: str, latitude: float,
                 longitude: float) -> None:
        self.stop_name = stop_name
        self.latitude = latitude
        self.longitude = longitude
    
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


class Graph:
    def __init__(self, nodes: dict[str, Node], edges: dict[str, list[Edge]] = None) -> None:
        self.edges = edges or {}
        self.nodes = nodes or {}
        
    def compare_times(self, current: datetime.time, departure: datetime.time) -> bool:
        return current <= departure
    
    def get_node(self, start_node: Node) -> Node:
        return self.nodes.get(start_node, None)
