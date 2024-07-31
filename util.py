import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ---- NetworkX compatibility
def node_iter(G):
    version = nx.__version__
    str_index = version.rfind('.')  # 获取最后一个点的下标
    short_version = version[:str_index]
    if float(short_version)<2.0:
        return G.nodes()
    else:
        return G.nodes

def node_dict(G):
    version = nx.__version__
    str_index = version.rfind('.')
    short_version = version[:str_index]
    if float(short_version)>2.1:
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict
# ---------------------------

def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, origin=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=arr.shape[::-1], dpi=1, frameon=False)
    canvas = FigureCanvas(fig)
    fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=1, format=format)

def plot_graph(plt, G, label):
    plt.clf()
    plt.title('num of nodes: '+str(G.number_of_nodes()), fontsize = 4)
    parts = community.best_partition(G)
    values = [parts.get(node) for node in G.nodes()]
    color_list = ['#bb12c6', '#224e1e', '#ee56ba', '#dca937', '#3536da', '#95df8e', '#9418c3', '#b2a8e4', '#05568d', '#27303d', '#fc8e84', '#2525ff', '#afb4ba', '#e0819f', '#8d7c46', '#52f3db', '#a61d9e', '#7f0bed', '#ca7f73', '#8da3dd', '#d7e0f5', '#5eb282', '#b58d8a', '#c790d8', '#4c7caf', '#10d9fa', '#c6355d', '#bde5a3', '#39fe1e', '#2c996e', '#83ad96', '#e2d0d0', '#da526e', '#7cccdd', '#4fd1a6', '#9ed268', '#23eb32', '#eb39cf', '#28c68e', '#1c0c39', '#f87ce1']

    colors = []
    colors_coarsen=[]
    grouped_graph = nx.Graph()
    for community_id in set(parts.values()):
        nodes_in_community = [nodes for nodes in parts.keys()
                              if parts[nodes] == community_id]
        first_node = nodes_in_community[0]
        colors_coarsen.append(color_list[community_id])
        grouped_graph.add_node('{}'.format(community_id))
    for u, v in G.edges():
        if parts[u] != parts[v]:
            group_u = parts[u]
            group_v = parts[v]
            if not grouped_graph.has_edge('{}'.format(group_u), '{}'.format(group_v)):
                grouped_graph.add_edge('{}'.format(group_u), '{}'.format(group_v))
    pos = nx.spring_layout(grouped_graph, seed=42)
    nx.draw(grouped_graph,
            pos=pos,
            with_labels=True, node_size=25, width=0.3, font_size = 3, node_color=colors_coarsen,)
    plt.savefig(str(G.number_of_nodes())+"coarsen", dpi=400)

    plt.clf()
    for i in range(len(values)):
        colors.append(color_list[values[i]])
    plt.axis("off")
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, with_labels=True, node_size=10, width=0.3, font_size=3, node_color=colors, pos=pos)
    plt.savefig(str(G.number_of_nodes())+"_"+str(label), dpi=400)

def draw_graph_list(G_list, row, col, fname = 'figs/test'):
    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row,col,i+1)
        plot_graph(plt, G)
        
    plt.tight_layout()
    plt.savefig(fname+'_view.png', dpi=600)
    plt.close()

    # draw degree distribution
    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i + 1)
        G_deg = np.array(list(G.degree(G.nodes()).values()))
        bins = np.arange(20)
        plt.hist(np.array(G_deg), bins=bins, align='left')
        plt.xlabel('degree', fontsize = 3)
        plt.ylabel('count', fontsize = 3)
        G_deg_mean = 2*G.number_of_edges()/float(G.number_of_nodes())
        plt.title('average degree: {:.2f}'.format(G_deg_mean), fontsize=4)
        plt.tick_params(axis='both', which='major', labelsize=3)
        plt.tick_params(axis='both', which='minor', labelsize=3)
    plt.tight_layout()
    plt.savefig(fname+'_degree.png', dpi=600)
    plt.close()


def exp_moving_avg(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a


