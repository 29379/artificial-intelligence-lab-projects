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
        
        df['departure_time'] = [time.time() for time in pd.to_datetime(df['departure_time'], format='%H:%M:%S')]
        df['arrival_time'] = [time.time() for time in pd.to_datetime(df['arrival_time'], format='%H:%M:%S')]

        nodes = {}
        edges = {}
        
        for row in df.itertuples():
            print(".")
            #   print(str(row.Index))
            start_node = Node(row.start_stop, float(row.start_stop_lat), float(row.start_stop_lon), [])
            end_node = Node(row.end_stop, float(row.end_stop_lat), float(row.end_stop_lon), [])

            departure_td = datetime.timedelta(hours=row.departure_time.hour, minutes=row.departure_time.minute, seconds=row.departure_time.second)
            arrival_td = datetime.timedelta(hours=row.arrival_time.hour, minutes=row.arrival_time.minute, seconds=row.arrival_time.second)

            edge = Edge(row.Index, row.company, start_node, end_node, row.line, row.departure_time, row.arrival_time)
            #edges.append(edge)
            if edge.start_node.stop_name not in edges:
                edges[edge.start_node.stop_name] = [edge]
            else:
                if edge not in edges[edge.start_node.stop_name]:
                    edges[edge.start_node.stop_name].append(edge)
            
            if edge not in start_node.outgoing_edges:
                start_node.outgoing_edges.append(edge)
            
            if start_node.stop_name not in nodes:
                nodes[start_node.stop_name] = start_node
            if end_node.stop_name not in nodes:
                nodes[end_node.stop_name] = end_node


        print("Data loaded")
        graph = Graph(nodes, edges)
        print("Graph created")
        return graph
    
