import os
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import torch

def convert_node_labels_to_integers(G, first_label=1, ordering="default", label_attribute=None):
    if ordering == "default":
        node_list = sorted(G.nodes())
    elif ordering == "degree":
        node_list = sorted(G.nodes(), key=G.degree)
    elif ordering == "random":
        node_list = list(G.nodes())
        random.shuffle(node_list)
    else:
        raise ValueError('Ordering not valid')

    if label_attribute is not None:
        mapping_dict = {node: i + first_label for i, node in enumerate(node_list)}
        nx.set_node_attributes(G, {str(mapping_dict[node]): G.nodes[node][label_attribute] for node in G.nodes()}, name=label_attribute)
    else:
        mapping_dict = {node: i + first_label for i, node in enumerate(node_list)}
    return nx.relabel_nodes(G, mapping_dict)

num_graphs_noncycle = 1000
num_graphs_cycle = 1000

os.makedirs('GraphCycle', exist_ok=True)
os.makedirs('GraphCycle/graphs', exist_ok=True)

graph_indicator_file = open('GraphCycle/graphcycle_graph_indicator.txt', 'w+')
graph_labels_file = open('GraphCycle/graphcycle_graph_labels.txt', 'w+')
node_labels_file = open('GraphCycle/graphcycle_node_labels.txt', 'w+')
edges_file = open('GraphCycle/graphcycle_A.txt', 'w+')
attributes_file = open('GraphCycle/graphcycle_node_attributes.txt', 'w+')

node_count = 0
edge_index_list = []
attributes_list = []
# edge_index_list_ground = []
# attributes_list_ground = []

for noncycle in range(num_graphs_noncycle):
    graphs = []
    num_graphs = 8
    mean = [0] * 10
    I = [[0 for _ in range(10)] for _ in range(10)]
    for i in range(10):
        I[i][i] = 3
    cov_c = I
    seeds = np.random.multivariate_normal(mean, cov_c, num_graphs)
    for i in range(num_graphs):
        n = random.randint(100, 150)
        G = nx.Graph()
        for j in range(n):
            label = f"{node_count}"
            G.add_node(label)
            node_count += 1
        I_1 = [[0 for _ in range(10)] for _ in range(10)]
        for k in range(10):
            I_1[k][k] = 1
        cov = I_1
        num_of_nodes = G.number_of_nodes()
        noise = np.random.multivariate_normal(mean, cov, num_of_nodes)
        index = 0
        for node in G.nodes():
            G.nodes[node]['features'] = (seeds[i]+noise[index]).tolist()
            index = index + 1
            node_labels_file.write(str(int(i) + 1) + '\n')
        for j in range(n):
            for k in range(n):
                if random.random() > 0.2:
                    if j != k:
                        label1 = f"{node_count-n+j}"
                        label2 = f"{node_count-n+k}"
                        G.add_edge(label1, label2)
        graphs.append(G)

    G = nx.Graph()
    for i in range(num_graphs):
        G = nx.compose(G, graphs[i])

    graph_size = G.number_of_nodes()
    for node in G.nodes():
        graph_indicator_file.write(str(noncycle+1) + '\n')

    G_small = nx.generators.random_tree(num_graphs)
    g_small_edge_list = list(G_small.edges)

    connected_pairs = set()

    for edge in range(len(g_small_edge_list)):
        source_index = g_small_edge_list[edge][0]
        source_graph = graphs[source_index]
        target_index = g_small_edge_list[edge][1]
        target_graph = graphs[target_index]
        num_conns = random.randint(1, 3)
        for j in range(num_conns):
            source_nodes = random.sample(list(source_graph.nodes()), random.randint(1, 2))
            target_nodes = random.sample(list(target_graph.nodes()), random.randint(1, 2))
            for node1 in source_nodes:
                node2 = random.choice(target_nodes)
                G.add_edge(node1, node2)

    G = convert_node_labels_to_integers(G)

    edge_index = torch.LongTensor(list(G.edges())).t().contiguous()
    edge_index[0] += node_count - graph_size
    edge_index[1] += node_count - graph_size
    edge_index_list.append(edge_index)


    graph_labels_file.write('0\n')

    node_attributes = nx.get_node_attributes(G, 'features')
    # node_attributes_ground = nx.get_node_attributes(G, 'features')
    for node, attributes in node_attributes.items():
        attributes_file.write(','.join(str(attr) for attr in attributes))
        attributes_file.write('\n')

    attributes_list.append(node_attributes)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.savefig(f"GraphCycle/graphs/graph_{noncycle + 1}.png")
    plt.clf()

for edge_index in edge_index_list:
    for i in range(edge_index.shape[1]):
        edges_file.write(str(edge_index[0][i].item()) + ', ' + str(edge_index[1][i].item()) + '\n')
        edges_file.write(str(edge_index[1][i].item()) + ', ' + str(edge_index[0][i].item()) + '\n')

# for edge_index in edge_index_list_ground:
#     for i in range(edge_index.shape[1]):
#         ground_edges_file.write(str(edge_index[0][i].item()) + ', ' + str(edge_index[1][i].item()) + '\n')
#         ground_edges_file.write(str(edge_index[1][i].item()) + ', ' + str(edge_index[0][i].item()) + '\n')

for cycle in range(num_graphs_cycle):
    graphs = []
    num_graphs = 8
    mean = [0] * 10
    I = [[0 for _ in range(10)] for _ in range(10)]
    for i in range(10):
        I[i][i] = 3
    cov_c = I
    seeds = np.random.multivariate_normal(mean, cov_c, num_graphs)
    for i in range(num_graphs):
        n = random.randint(100, 150)
        G = nx.Graph()
        for j in range(n):
            label = f"{node_count}"
            G.add_node(label)
            node_count += 1
        # seed = np.random.uniform(low=i + 1, high=i + 2, size=(1, 20))
        num_of_nodes = G.number_of_nodes()
        I_1 = [[0 for _ in range(10)] for _ in range(10)]
        for k in range(10):
            I_1[k][k] = 1
        cov = I_1
        noise = np.random.multivariate_normal(mean, cov, num_of_nodes)
        index = 0
        for node in G.nodes():
            G.nodes[node]['features'] = (seeds[i]+noise[index]).tolist()
            index = index + 1
            node_labels_file.write(str(int(i) + 1) + '\n')
        for j in range(n):
            for k in range(n):
                if random.random() > 0.2:
                    if j != k:
                        label1 = f"{node_count-n+j}"
                        label2 = f"{node_count-n+k}"
                        G.add_edge(label1, label2)
        graphs.append(G)

    G = nx.Graph()
    for i in range(num_graphs):
        G = nx.compose(G, graphs[i])

    graph_size = G.number_of_nodes()

    for node in G.nodes():
        graph_indicator_file.write(str(num_graphs_noncycle + cycle + 1) + '\n')

    cycle_nodes = []
    cycle_edges = []
    cycle_len = np.random.randint(3, num_graphs + 1)
    cycle_nodes = np.random.choice(num_graphs, cycle_len, replace=False).tolist()
    cycle_edges = [(cycle_nodes[i], cycle_nodes[(i + 1) % cycle_len]) for i in range(cycle_len)]

    G_small = nx.Graph()
    G_small.add_nodes_from(range(num_graphs))
    G_small.add_edges_from(cycle_edges)
    isolated_nodes = list(nx.isolates(G_small))
    len_cycle = len(cycle_nodes)
    for iso_node in isolated_nodes:
        connected_node = np.random.randint(len_cycle)
        G_small.add_edge(iso_node, cycle_nodes[connected_node])

    g_small_edge_list = list(G_small.edges)
    connected_pairs = set()

    for edge in range(len(g_small_edge_list)):
        source_index = g_small_edge_list[edge][0]
        source_graph = graphs[source_index]
        target_index = g_small_edge_list[edge][1]
        target_graph = graphs[target_index]
        num_conns = random.randint(1, 3)
        for j in range(num_conns):
            source_nodes = random.sample(list(source_graph.nodes()), random.randint(1, 2))
            target_nodes = random.sample(list(target_graph.nodes()), random.randint(1, 2))
            for node1 in source_nodes:
                node2 = random.choice(target_nodes)
                G.add_edge(node1, node2)

    G = convert_node_labels_to_integers(G)
    edge_index = torch.LongTensor(list(G.edges())).t().contiguous()
    edge_index[0] += node_count - graph_size
    edge_index[1] += node_count - graph_size
    edge_index_list.append(edge_index)
    graph_labels_file.write('1\n')

    node_attributes = nx.get_node_attributes(G, 'features')
    for node, attributes in node_attributes.items():
        attributes_file.write(','.join(str(attr) for attr in attributes))
        attributes_file.write('\n')
    attributes_list.append(node_attributes)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.savefig(f"GraphCycle/graphs/graph_{num_graphs_noncycle + cycle + 1}.png")
    plt.clf()

for edge_index in edge_index_list:
    for i in range(edge_index.shape[1]):
        edges_file.write(str(edge_index[0][i].item()) + ', ' + str(edge_index[1][i].item()) + '\n')
        edges_file.write(str(edge_index[1][i].item()) + ', ' + str(edge_index[0][i].item()) + '\n')

graph_indicator_file.close()
graph_labels_file.close()
edges_file.close()
attributes_file.close()