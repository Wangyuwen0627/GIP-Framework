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

num_graphs_caveman_graph = 1000
num_graphs_cycle_graph = 1000
num_graphs_grid_graph = 1000
num_graphs_ladder_graph = 1000
num_graphs_star_graph = 1000

os.makedirs('FiveClass', exist_ok=True)
os.makedirs('FiveClass/graphs', exist_ok=True)

graph_indicator_file = open('FiveClass/fiveclass_graph_indicator.txt', 'w+')
graph_labels_file = open('FiveClass/fiveclass_graph_labels.txt', 'w+')
node_labels_file = open('FiveClass/fiveclass_node_labels.txt', 'w+')
edges_file = open('FiveClass/fiveclass_A.txt', 'w+')
attributes_file = open('FiveClass/fiveclass_node_attributes.txt', 'w+')

node_count = 0

edge_index_list = []
attributes_list = []

print("==================generate caveman class======================")
for caveman in range(num_graphs_caveman_graph):
    graphs = []
    num_graphs = random.randint(12, 16)
    mean = [0] * 10
    I = [[0 for _ in range(10)] for _ in range(10)]
    for k in range(10):
        I[k][k] = 3
    cov_c = I
    seeds = np.random.multivariate_normal(mean, cov_c, num_graphs)
    for i in range(num_graphs):
        n = random.randint(10, 15)
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
                if random.random() > 0.1:
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
        graph_indicator_file.write(str(caveman + 1) + '\n')

    G_small = nx.wheel_graph(num_graphs)

    g_small_edge_list = list(G_small.edges)
    connected_pairs = set()

    for edge in range(len(g_small_edge_list)):
        source_index = g_small_edge_list[edge][0]
        source_graph = graphs[source_index]
        target_index = g_small_edge_list[edge][1]
        target_graph = graphs[target_index]
        source_nodes = random.sample(list(source_graph.nodes()), 1)
        target_nodes = random.sample(list(target_graph.nodes()), 1)
        G.add_edge(source_nodes[0], target_nodes[0])
    G = convert_node_labels_to_integers(G)
    edge_index = torch.LongTensor(list(G.edges())).t().contiguous()
    edge_index[0] += node_count - graph_size
    edge_index[1] += node_count - graph_size
    edge_index_list.append(edge_index)

    graph_labels_file.write('0\n')

    node_attributes = nx.get_node_attributes(G, 'features')
    for node, attributes in node_attributes.items():
        attributes_file.write(','.join(str(attr) for attr in attributes))
        attributes_file.write('\n')
    attributes_list.append(node_attributes)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.savefig(f"FiveClass/graphs/graph_{caveman + 1}.png")
    plt.clf()

for edge_index in edge_index_list:
    for i in range(edge_index.shape[1]):
        edges_file.write(str(edge_index[0][i].item()) + ', ' + str(edge_index[1][i].item()) + '\n')
        edges_file.write(str(edge_index[1][i].item()) + ', ' + str(edge_index[0][i].item()) + '\n')

print("==================generate cycle class======================")
for cycle in range(num_graphs_cycle_graph):
    graphs = []
    num_graphs = random.randint(12, 16)
    mean = [0] * 10
    I = [[0 for _ in range(10)] for _ in range(10)]
    for k in range(10):
        I[k][k] = 3
    cov_c = I
    seeds = np.random.multivariate_normal(mean, cov_c, num_graphs)
    for i in range(num_graphs):
        n = random.randint(10, 15)
        G = nx.Graph()
        for j in range(n):
            label = f"{node_count}"
            G.add_node(label)
            node_count += 1
        # seed = np.random.rand(1, 20)
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
                if random.random() > 0.1:
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
        graph_indicator_file.write(str(num_graphs_caveman_graph + cycle + 1) + '\n')

    G_small = nx.cycle_graph(num_graphs)

    g_small_edge_list = list(G_small.edges)
    connected_pairs = set()

    for edge in range(len(g_small_edge_list)):
        source_index = g_small_edge_list[edge][0]
        source_graph = graphs[source_index]
        target_index = g_small_edge_list[edge][1]
        target_graph = graphs[target_index]
        source_nodes = random.sample(list(source_graph.nodes()), 1)
        target_nodes = random.sample(list(target_graph.nodes()), 1)
        G.add_edge(source_nodes[0], target_nodes[0])

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
    plt.savefig(f"FiveClass/graphs/graph_{num_graphs_caveman_graph + cycle + 1}.png")
    plt.clf()

for edge_index in edge_index_list:
    for i in range(edge_index.shape[1]):
        edges_file.write(str(edge_index[0][i].item()) + ', ' + str(edge_index[1][i].item()) + '\n')
        edges_file.write(str(edge_index[1][i].item()) + ', ' + str(edge_index[0][i].item()) + '\n')

print("==================generate grid class======================")
for grid in range(num_graphs_grid_graph):
    graphs = []
    num_graphs = random.randint(12, 16)
    mean = [0] * 10
    I = [[0 for _ in range(10)] for _ in range(10)]
    for k in range(10):
        I[k][k] = 3
    cov_c = I
    seeds = np.random.multivariate_normal(mean, cov_c, num_graphs)
    for i in range(num_graphs):
        n = random.randint(10, 15)
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
            node_labels_file.write(str(int(i) + 1) + '\n')
        for j in range(n):
            for k in range(n):
                if random.random() > 0.1:
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
        graph_indicator_file.write(str(num_graphs_caveman_graph + num_graphs_cycle_graph + grid + 1) + '\n')

    d1 = 4
    d2 = int(num_graphs / d1)
    G_small = nx.grid_graph(dim=[d1, d2])
    new_labels = []

    for node in G_small.nodes():
        new_label = node[0] * d1 + node[1]
        new_labels.append(str(new_label))
    my_dict = dict(zip(G_small.nodes(), new_labels))
    nx.relabel_nodes(G_small, my_dict, copy=False)

    remains = num_graphs - d1 * d2
    for remain in range(remains):
        G_small.add_node(str(d1 * d2 + remain))
        connect_target = random.randint(0, (d1 * d2-1))
        G_small.add_edge(str(connect_target), str(d1 * d2 + remain))

    g_small_edge_list = list(G_small.edges)
    connected_pairs = set()
    for edge in range(len(g_small_edge_list)):
        source_index = g_small_edge_list[edge][0]
        source_graph = graphs[int(source_index)]
        target_index = g_small_edge_list[edge][1]
        target_graph = graphs[int(target_index)]
        source_nodes = random.sample(list(source_graph.nodes()), 1)
        target_nodes = random.sample(list(target_graph.nodes()), 1)
        G.add_edge(source_nodes[0], target_nodes[0])

    G = convert_node_labels_to_integers(G)
    edge_index = torch.LongTensor(list(G.edges())).t().contiguous()
    edge_index[0] += node_count - graph_size
    edge_index[1] += node_count - graph_size
    edge_index_list.append(edge_index)

    graph_labels_file.write('2\n')

    node_attributes = nx.get_node_attributes(G, 'features')
    for node, attributes in node_attributes.items():
        attributes_file.write(','.join(str(attr) for attr in attributes))
        attributes_file.write('\n')
    attributes_list.append(node_attributes)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.savefig(f"FiveClass/graphs/graph_{num_graphs_caveman_graph + num_graphs_cycle_graph + grid + 1}.png")
    plt.clf()

for edge_index in edge_index_list:
    for i in range(edge_index.shape[1]):
        edges_file.write(str(edge_index[0][i].item()) + ', ' + str(edge_index[1][i].item()) + '\n')
        edges_file.write(str(edge_index[1][i].item()) + ', ' + str(edge_index[0][i].item()) + '\n')

print("==================generate ladder class======================")
for ladder in range(num_graphs_ladder_graph):
    graphs = []
    num_graphs = random.randint(12, 16)
    mean = [0] * 10
    I = [[0 for _ in range(10)] for _ in range(10)]
    for k in range(10):
        I[k][k] = 3
    cov_c = I
    seeds = np.random.multivariate_normal(mean, cov_c, num_graphs)
    for i in range(num_graphs):
        n = random.randint(10, 15)
        G = nx.Graph()
        for j in range(n):
            label = f"{node_count}"
            G.add_node(label)
            node_count += 1
        # seed = np.random.rand(1, 20)
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
                if random.random() > 0.1:
                    if j != k:
                        label1 = f"{node_count - n + j}"
                        label2 = f"{node_count - n + k}"
                        G.add_edge(label1, label2)
        graphs.append(G)

    G = nx.Graph()
    for i in range(num_graphs):
        G = nx.compose(G, graphs[i])
    graph_size = G.number_of_nodes()

    for node in G.nodes():
        graph_indicator_file.write(str(num_graphs_caveman_graph + num_graphs_cycle_graph + num_graphs_grid_graph + ladder + 1) + '\n')

    layer = int(num_graphs/2)
    G_small = nx.ladder_graph(layer)
    remains = num_graphs - layer * 2
    for remain in range(remains):
        G_small.add_node(str(layer * 2 + remain))
        connect_target = random.randint(0, (layer * 2-1))
        G_small.add_edge(str(connect_target), str(layer * 2 + remain))

    g_small_edge_list = list(G_small.edges)
    connected_pairs = set()
    for edge in range(len(g_small_edge_list)):
        source_index = g_small_edge_list[edge][0]
        source_graph = graphs[int(source_index)]
        target_index = g_small_edge_list[edge][1]
        target_graph = graphs[int(target_index)]
        source_nodes = random.sample(list(source_graph.nodes()), 1)
        target_nodes = random.sample(list(target_graph.nodes()), 1)
        G.add_edge(source_nodes[0], target_nodes[0])

    G = convert_node_labels_to_integers(G)
    edge_index = torch.LongTensor(list(G.edges())).t().contiguous()
    edge_index[0] += node_count - graph_size
    edge_index[1] += node_count - graph_size
    edge_index_list.append(edge_index)

    graph_labels_file.write('3\n')

    node_attributes = nx.get_node_attributes(G, 'features')
    for node, attributes in node_attributes.items():
        attributes_file.write(','.join(str(attr) for attr in attributes))
        attributes_file.write('\n')
    attributes_list.append(node_attributes)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.savefig(f"FiveClass/graphs/graph_{num_graphs_caveman_graph + num_graphs_cycle_graph + num_graphs_grid_graph + ladder + 1}.png")
    plt.clf()

for edge_index in edge_index_list:
    for i in range(edge_index.shape[1]):
        edges_file.write(str(edge_index[0][i].item()) + ', ' + str(edge_index[1][i].item()) + '\n')
        edges_file.write(str(edge_index[1][i].item()) + ', ' + str(edge_index[0][i].item()) + '\n')


print("==================generate star class======================")

for star in range(num_graphs_star_graph):
    graphs = []
    num_graphs = random.randint(12, 16)
    mean = [0] * 10
    I = [[0 for _ in range(10)] for _ in range(10)]
    for k in range(10):
        I[k][k] = 3
    cov_c = I
    seeds = np.random.multivariate_normal(mean, cov_c, num_graphs)
    for i in range(num_graphs):
        n = random.randint(10, 15)
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
                if random.random() > 0.1:
                    if j != k:
                        label1 = f"{node_count - n + j}"
                        label2 = f"{node_count - n + k}"
                        G.add_edge(label1, label2)
        graphs.append(G)

    G = nx.Graph()
    for i in range(num_graphs):
        G = nx.compose(G, graphs[i])

    graph_size = G.number_of_nodes()

    for node in G.nodes():
        graph_indicator_file.write(str(num_graphs_caveman_graph + num_graphs_cycle_graph + num_graphs_grid_graph + num_graphs_ladder_graph + star + 1) + '\n')

    G_small = nx.star_graph(num_graphs-1)
    g_small_edge_list = list(G_small.edges)

    connected_pairs = set()

    for edge in range(len(g_small_edge_list)):
        source_index = g_small_edge_list[edge][0]
        source_graph = graphs[source_index]
        target_index = g_small_edge_list[edge][1]
        target_graph = graphs[target_index]
        source_nodes = random.sample(list(source_graph.nodes()), 1)
        target_nodes = random.sample(list(target_graph.nodes()), 1)
        G.add_edge(source_nodes[0], target_nodes[0])

    G = convert_node_labels_to_integers(G)
    edge_index = torch.LongTensor(list(G.edges())).t().contiguous()
    edge_index[0] += node_count - graph_size
    edge_index[1] += node_count - graph_size
    edge_index_list.append(edge_index)

    graph_labels_file.write('4\n')

    node_attributes = nx.get_node_attributes(G, 'features')
    for node, attributes in node_attributes.items():
        attributes_file.write(','.join(str(attr) for attr in attributes))
        attributes_file.write('\n')
    attributes_list.append(node_attributes)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.savefig(f"FiveClass/graphs/graph_{num_graphs_caveman_graph + num_graphs_cycle_graph + num_graphs_grid_graph + num_graphs_ladder_graph + star + 1}.png")
    plt.clf()

for edge_index in edge_index_list:
    for i in range(edge_index.shape[1]):
        edges_file.write(str(edge_index[0][i].item()) + ', ' + str(edge_index[1][i].item()) + '\n')
        edges_file.write(str(edge_index[1][i].item()) + ', ' + str(edge_index[0][i].item()) + '\n')

graph_indicator_file.close()
graph_labels_file.close()
edges_file.close()
attributes_file.close()