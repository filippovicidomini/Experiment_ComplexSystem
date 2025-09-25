
# networkx_sims.py
# Pure NetworkX simulations: SIR, SIS, percolation, robustness, ER vs BA helpers.
# Requirements: networkx, matplotlib. (pandas only if you want tables.)

import random
import math
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------
# Loaders / generators
# -------------------------
def load_karate():
    return nx.karate_club_graph()

def gen_er(n=100, p=0.03, seed=42):
    """Generate an Erdős-Rényi random graph.

    Args:
        n (int, optional): Number of nodes. Defaults to 100.
        p (float, optional): Probability of edge creation. Defaults to 0.03.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        networkx.Graph: Generated Erdős-Rényi graph.
    """
    return nx.gnp_random_graph(n, p, seed=seed, directed=False)


def gen_ba(n=100, m=2, seed=42):
    """Generate a Barabási-Albert scale-free network.
    Args:
        n (int, optional): Number of nodes. Defaults to 100.
        m (int, optional): Number of edges to attach from a new node to existing nodes. Defaults to 2.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    Returns:
        networkx.Graph: Generated Barabási-Albert graph.
    """
    return nx.barabasi_albert_graph(n, m, seed=seed)

# -------------------------
# Basic analysis helpers
# -------------------------
def basic_metrics(G):
    info = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'avg_clustering': nx.average_clustering(G),
        'assortativity_deg': nx.degree_assortativity_coefficient(G),
        'connected': nx.is_connected(G),
    }
    H = G if nx.is_connected(G) else G.subgraph(max(nx.connected_components(G), key=len)).copy()
    info['avg_path_length'] = nx.average_shortest_path_length(H)
    info['diameter'] = nx.diameter(H)
    return info

def plot_degree_distribution(G):
    degs = [d for _, d in G.degree()]
    plt.figure()
    plt.hist(degs, bins=range(0, max(degs)+2), align='left', rwidth=0.8)
    plt.title("Degree distribution")
    plt.xlabel("Degree k")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def draw_network(G, seed=42, with_labels=True, scale_by_degree=True):
    pos = nx.spring_layout(G, seed=seed)
    if scale_by_degree:
        sizes = [100 + 40*d for _, d in G.degree()]
    else:
        sizes = 300
    plt.figure()
    nx.draw_networkx(G, pos=pos, with_labels=with_labels, node_size=sizes, font_size=8)
    plt.title("Network")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------
# Epidemic models on graphs
# -------------------------
# States: 0=S, 1=I, 2=R
def sir(G, beta=0.08, gamma=0.05, initial_infected=None, steps=200, seed=123):
    random.seed(seed)
    state = {u:0 for u in G.nodes()}
    if initial_infected is None:
        initial_infected = max(G.degree, key=lambda x: x[1])[0]
    state[initial_infected] = 1

    S, I, R = [], [], []
    for t in range(steps):
        S.append(sum(1 for v in state.values() if v==0))
        I.append(sum(1 for v in state.values() if v==1))
        R.append(sum(1 for v in state.values() if v==2))
        if I[-1] == 0:
            break
        new_state = state.copy()
        for u in G.nodes():
            if state[u] == 1:
                for v in G.neighbors(u):
                    if state[v] == 0 and random.random() < beta:
                        new_state[v] = 1
        for u in G.nodes():
            if state[u] == 1 and random.random() < gamma:
                new_state[u] = 2
        state = new_state
    return {'S': S, 'I': I, 'R': R}

# States: 0=S, 1=I (no recovered), mu recovery prob -> back to S
def sis(G, beta=0.08, mu=0.05, initial_infected=None, steps=200, seed=123):
    random.seed(seed)
    state = {u:0 for u in G.nodes()}
    if initial_infected is None:
        initial_infected = max(G.degree, key=lambda x: x[1])[0]
    state[initial_infected] = 1

    S, I = [], []
    for t in range(steps):
        S.append(sum(1 for v in state.values() if v==0))
        I.append(sum(1 for v in state.values() if v==1))
        if I[-1] == 0:
            break
        new_state = state.copy()
        for u in G.nodes():
            if state[u] == 1:
                for v in G.neighbors(u):
                    if state[v] == 0 and random.random() < beta:
                        new_state[v] = 1
        for u in G.nodes():
            if state[u] == 1 and random.random() < mu:
                new_state[u] = 0
        state = new_state
    return {'S': S, 'I': I}

def plot_curves(curves, title="Epidemic curves"):
    plt.figure()
    for key, series in curves.items():
        plt.plot(series, label=key)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Number of nodes")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------
# Robustness / percolation
# -------------------------
def random_failures_gcc(G, frac=0.1, trials=20, seed=42):
    # remove a fraction of nodes uniformly at random; return average GCC size
    random.seed(seed)
    n = G.number_of_nodes()
    k = int(frac * n)
    sizes = []
    for _ in range(trials):
        removed = random.sample(list(G.nodes()), k)
        H = G.copy()
        H.remove_nodes_from(removed)
        if H.number_of_nodes() == 0:
            sizes.append(0)
        else:
            GCC = max(nx.connected_components(H), key=len) if H.number_of_nodes() > 0 else set()
            sizes.append(len(GCC) if GCC else 0)
    return sum(sizes) / len(sizes)

def targeted_attacks_gcc(G, frac=0.1):
    # remove the top-k by degree
    n = G.number_of_nodes()
    k = int(frac * n)
    by_deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    to_remove = [u for u,_ in by_deg[:k]]
    H = G.copy()
    H.remove_nodes_from(to_remove)
    if H.number_of_nodes() == 0:
        return 0
    GCC = max(nx.connected_components(H), key=len) if H.number_of_nodes() > 0 else set()
    return len(GCC) if GCC else 0

def plot_robustness(G, steps=10, trials=30):
    fracs = [i/steps for i in range(steps+1)]
    rf = [random_failures_gcc(G, f, trials=trials, seed=123) for f in fracs]
    ta = [targeted_attacks_gcc(G, f) for f in fracs]
    plt.figure()
    plt.plot(fracs, rf, marker='o', label='Random failures (avg GCC)')
    plt.plot(fracs, ta, marker='s', label='Targeted attacks (GCC)')
    plt.title("Robustness: GCC size vs removed fraction")
    plt.xlabel("Removed node fraction")
    plt.ylabel("GCC size")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------
# Demo
# -------------------------
if __name__ == '__main__':
    G = load_karate()
    print('Basic metrics:', basic_metrics(G))
    plot_degree_distribution(G)
    draw_network(G)
    curves = sir(G, beta=0.08, gamma=0.05, seed=7)
    plot_curves(curves, title='SIR on Karate Club')
    plot_robustness(G, steps=10, trials=50)
