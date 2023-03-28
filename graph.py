from datetime import datetime, timedelta
import heapq

# class PriorityQueue:
#     def __init__(self) -> None:
#         self.pq: list[tuple[float, Node]] = []

#     def is_empty(self) -> bool:
#         return not self.pq

#     def enqueue(self, item: 'Node', priority: float) -> None:
#         heapq.heappush(self.pq, (priority, item))

#     def dequeue(self) -> 'Node':
#         return heapq.heappop(self.pq)[1]

#     def contains(self, node: 'Node') -> bool:
#         for priority, elem in self.pq:
#             if elem == node:
#                 return True
#         return False

# class PriorityQueue:
#     def __init__(self) -> None:
#         self.pq: list[Node] = []

#     def is_empty(self) -> bool:
#         return len(self.pq) == 0

#     def enqueue(self, item: 'Node', priority: float):
#         heapq.heappush(self.pq, (priority, item))

#     def dequeue(self) -> 'Node':
#         return heapq.heappop(self.pq)[1]

#     def contains(self, item: 'Node') -> bool:
#         for _, element in self.pq:
#             if element == item:
#                 return True
#         return False
    
#     def update_priority(self, item: 'Node', new_priority: float) -> None:
#         for i, (priority, element) in enumerate(self.pq):
#             if element == item:
#                 self.pq[i] = (new_priority, element)
#                 heapq.heapify(self.pq)
#                 break

class PriorityQueue:
    def __init__(self) -> None:
        self.pq: list[tuple[float, 'Node']] = []
        self.item_map: dict['Node', float] = {}

    def is_empty(self) -> bool:
        return len(self.pq) == 0

    def enqueue(self, item: 'Node', priority: float):
        heapq.heappush(self.pq, (priority, item))
        self.item_map[item] = len(self.pq) - 1

    def dequeue(self) -> 'Node':
        priority, item = heapq.heappop(self.pq)
        del self.item_map[item]
        return item

    def contains(self, item: 'Node') -> bool:
        return item in self.item_map

    def update_priority(self, item: 'Node', new_priority: float) -> None:
        if item not in self.item_map:
            raise ValueError("Item not found in priority queue")
        i = self.item_map[item]
        _, item = self.pq[i]
        self.pq[i] = (new_priority, item)
        self.item_map[item] = i
        heapq._siftup(self.pq, i)


class Node:
    def __init__(self, stop_name: str, latitude: float,
                 longitude: float, outgoing_edges: list['Edge']) -> None:
        self.stop_name = stop_name
        self.latitude = latitude
        self.longitude = longitude
        self.outgoing_edges = outgoing_edges
        
    def add_edge(self, edge: 'Edge') -> None:
        self.outgoing_edges.append(edge)
    
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
        self.departure_time = datetime.strptime(departure_time.strftime('%H:%M:%S'), '%H:%M:%S').time()
        self.arrival_time = datetime.strptime(arrival_time.strftime('%H:%M:%S'), '%H:%M:%S').time()
        
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
    def __init__(self, nodes: dict[str, Node] = None, 
                    edges: dict[str, list[Edge]] = None) -> None:
        self.nodes = nodes or {}
        self.edges = edges or {}
        
    def compare_times(self, current: datetime.time, departure: datetime.time) -> bool:
        return current <= departure
        
    def get_node(self, name: str) -> Node:
        return self.nodes.get(name, None)
            
    def get_outgoing_edges(self, start_node: Node) -> list[Edge]:
        return self.edges.get(start_node.stop_name, [])
    
    def get_neighbor_nodes(self, node:Node) -> list[Node]:
        adjacent_edges = self.get_outgoing_edges(node)
        neighbor_nodes = []
        for edge in adjacent_edges:
            neighbor_nodes.append(edge.end_node)
        return neighbor_nodes
    
    def add_node(self, node: Node) -> None:
        if node.stop_name in self.nodes:
            for edge in node.outgoing_edges:
                if edge not in self.nodes[node.stop_name]:
                    self.nodes[node.stop_name].append(edge)
        else:
            self.nodes[node.stop_name] = node
    
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


