import pandas as pd
from graph import *
import os
import datetime

class DataReader():
    @staticmethod
    def load_data() -> Graph:
        filepath = "D:/University Projects/SI/Lista1/connection_graph.csv"
        df = pd.read_csv(filepath, encoding='utf-8', sep=',', skiprows=[0],
            dtype={"id": str, 
                "company": str, 
                "line": str, 
                "departure_time": str, 
                "arrival_time": str, 
                "start_stop": str, 
                "end_stop": str, 
                "start_stop_lat": str, 
                "start_stop_lon": str, 
                "end_stop_lat": str, 
                "end_stop_lon": str},
            names=["id", "company", "line", "departure_time",
                   "arrival_time", "start_stop", "end_stop",
                   "start_stop_lat", "start_stop_lon",
                   "end_stop_lat", "end_stop_lon"],
            parse_dates=['departure_time', 'arrival_time'])
        
        # df['departure_time'] = [time.time() for time in pd.to_datetime(df['departure_time'], format='%H:%M:%S')]
        # df['arrival_time'] = [time.time() for time in pd.to_datetime(df['arrival_time'], format='%H:%M:%S')]

        df['departure_time'] = pd.to_datetime(df['departure_time'], format='%H:%M:%S').dt.time
        df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S').dt.time
        
        #   max_arrival_time = pd.to_datetime(df['arrival_time'], format='%H:%M:%S').max().strftime('%H:%M:%S')
        #   print(max_arrival_time)

        nodes = {}
        edges = {}
        
        for row in df.itertuples():
            start_node = Node(row.start_stop, float(row.start_stop_lat), float(row.start_stop_lon), [])
            end_node = Node(row.end_stop, float(row.end_stop_lat), float(row.end_stop_lon), [])

            departure_td = datetime.timedelta(hours=row.departure_time.hour, minutes=row.departure_time.minute, seconds=row.departure_time.second)
            arrival_td = datetime.timedelta(hours=row.arrival_time.hour, minutes=row.arrival_time.minute, seconds=row.arrival_time.second)

            edge = Edge(row.Index, row.company, start_node, end_node, row.line, departure_td, arrival_td)
            
            if edge.start_node.stop_name not in edges:
                edges[edge.start_node.stop_name] = [edge]
            else:
                edges[edge.start_node.stop_name].append(edge)
            
            if edge not in start_node.outgoing_edges:
                start_node.outgoing_edges.append(edge)
            
            if start_node.stop_name not in nodes:
                nodes[start_node.stop_name] = start_node
            if end_node.stop_name not in nodes:
                nodes[end_node.stop_name] = end_node
        print("Data loaded")
        print(f"Number of vertices: {len(edges)} / {len(nodes)}")
        print(f"Number of edges: {sum(len(lst) for lst in edges.values())}")
        graph = Graph(nodes, edges)
        print("Graph created")
        print("\n- - - - - - - - - - - - - - - - - - - -\n")
        return graph
    
