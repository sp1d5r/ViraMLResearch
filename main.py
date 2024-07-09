import onnx
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import netron

class ModelWrapper:
    def __init__(self, onnx_path):
        self.model = onnx.load(onnx_path)
        self.graph = self.model.graph

    def print_model_info(self):
        # Print information about the model
        print(f"Model inputs: {[input.name for input in self.graph.input]}")
        print(f"Model outputs: {[output.name for output in self.graph.output]}")
        # Loop through the nodes and print some details, customize as needed
        for node in self.graph.node:
            print(f"Node name: {node.name}")
            print(f"Node type: {node.op_type}")
            print(f"Node inputs: {node.input}")
            print(f"Node outputs: {node.output}")
            print("----------")

    def to_networkx(self):
        G = nx.DiGraph()
        for node in self.graph.node:
            G.add_node(node.name, op_type=node.op_type)
            for inp in node.input:
                G.add_edge(inp, node.name)
            for out in node.output:
                G.add_edge(node.name, out)
        return G

    def visualize_graph(self, G):
        pos = graphviz_layout(G, prog='dot')  # hierarchical layout
        plt.figure(figsize=(16, 16))  # Increase figure size

        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
        nx.draw_networkx_labels(G, pos, font_size=8)  # Reduce font size and rotate labels

        plt.title("ONNX Model Graph")
        plt.show()

    def partition_and_visualize(self, G, partitions):
        pos = graphviz_layout(G, prog='dot')  # hierarchical layout
        plt.figure(figsize=(16, 16))  # Increase figure size

        color_map = plt.get_cmap('viridis')
        colors = [color_map(i / partitions) for i in range(partitions)]

        subgraphs = list(nx.connected_components(G.to_undirected()))
        for i, subgraph in enumerate(subgraphs[:partitions]):
            nx.draw_networkx_nodes(G, pos, nodelist=subgraph, node_size=500, node_color=[colors[i]])
            nx.draw_networkx_edges(G, pos, edgelist=G.edges(subgraph), edge_color=colors[i])

        nx.draw_networkx_labels(G, pos, font_size=8)  # Reduce font size and rotate labels

        plt.title(f"ONNX Model Graph Partitioned into {partitions} Parts")
        plt.show()

    def launch_netron(self):
        netron.start(self.model.graph)

if __name__ == '__main__':
    model_path = 'example_onnx_files/model.onnx'
    wrapper = ModelWrapper(model_path)
    wrapper.print_model_info()

    # G = wrapper.to_networkx()
    # wrapper.visualize_graph(G)
    #
    # partitions = 3  # Example: partition into 3 parts
    # wrapper.partition_and_visualize(G, partitions)

    # Launch Netron for graphical visualization
    netron.start(model_path)