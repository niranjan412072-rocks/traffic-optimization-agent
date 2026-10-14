"""Congestion-aware route recommendation using A* on a city grid graph.

Edge travel time follows the standard BPR volume-delay function:
    t = t_free * (1 + 0.15 * (volume/capacity) ** 4)

Run:  python -m src.routing
"""
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

FREE_SPEED_KMH = 40.0


def build_city_graph(n=8, seed=config.SEED):
    rng = np.random.default_rng(seed)
    G = nx.grid_2d_graph(n, n)
    for node in G.nodes:
        G.nodes[node]["pos"] = (node[0] * 0.5, node[1] * 0.5)      # km
    for u, v in G.edges:
        pu, pv = np.array(G.nodes[u]["pos"]), np.array(G.nodes[v]["pos"])
        length = float(np.linalg.norm(pu - pv)) * (1 + rng.uniform(0, 0.2))   # >= straight line
        G.edges[u, v]["length"] = length
        G.edges[u, v]["t_free"] = length / FREE_SPEED_KMH * 60                # minutes
        G.edges[u, v]["vc"] = 0.0
        G.edges[u, v]["time_min"] = G.edges[u, v]["t_free"]
    return G


def apply_congestion(G, vc_by_edge):
    """vc_by_edge: {(u, v): volume/capacity ratio}."""
    for (u, v), vc in vc_by_edge.items():
        e = G.edges[u, v]
        e["vc"] = float(vc)
        e["time_min"] = e["t_free"] * (1 + 0.15 * vc ** 4)
    return G


def hotspot_congestion(G, centre=None, strength=2.2, sigma=1.1, seed=config.SEED):
    """Congestion concentrated around a hotspot (e.g. predicted by the forecaster)."""
    rng = np.random.default_rng(seed)
    xs = [p[0] for p in nx.get_node_attributes(G, "pos").values()]
    centre = centre or (max(xs) / 2, max(xs) / 2)
    out = {}
    for u, v in G.edges:
        mid = (np.array(G.nodes[u]["pos"]) + np.array(G.nodes[v]["pos"])) / 2
        d = np.linalg.norm(mid - np.array(centre))
        out[(u, v)] = float(np.clip(strength * np.exp(-d ** 2 / (2 * sigma ** 2)) + rng.normal(0, 0.05) + 0.2, 0.05, 2.5))
    return out


def _heuristic(G):
    def h(a, b):
        d = np.linalg.norm(np.array(G.nodes[a]["pos"]) - np.array(G.nodes[b]["pos"]))
        return d / FREE_SPEED_KMH * 60          # admissible: never over-estimates
    return h


def fastest_route(G, src, dst):
    return nx.astar_path(G, src, dst, heuristic=_heuristic(G), weight="time_min")


def shortest_distance_route(G, src, dst):
    return nx.shortest_path(G, src, dst, weight="length")


def route_time(G, path):
    return sum(G.edges[a, b]["time_min"] for a, b in zip(path[:-1], path[1:]))


def evaluate_routing(G, n_pairs=300, seed=config.SEED):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes)
    saved, rows = [], []
    for _ in range(n_pairs):
        s, t = rng.choice(len(nodes), 2, replace=False)
        s, t = nodes[s], nodes[t]
        p_dist, p_fast = shortest_distance_route(G, s, t), fastest_route(G, s, t)
        td, tf = route_time(G, p_dist), route_time(G, p_fast)
        saved.append(100 * (td - tf) / td)
        rows.append((td, tf))
    rows = np.array(rows)
    return {"pairs": n_pairs, "mean_time_saved_%": float(np.mean(saved)),
            "median_time_saved_%": float(np.median(saved)),
            "max_time_saved_%": float(np.max(saved)),
            "mean_time_distance_route_min": float(rows[:, 0].mean()),
            "mean_time_ai_route_min": float(rows[:, 1].mean())}


def plot_routes(G, src, dst, path_a, path_b, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    pos = nx.get_node_attributes(G, "pos")
    vc = [G.edges[e]["vc"] for e in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=vc, edge_cmap=plt.cm.RdYlGn_r,
                           edge_vmin=0, edge_vmax=2.0, width=3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=8, node_color="#444")
    for path, col, lab in ((path_a, "royalblue", "Shortest distance"), (path_b, "black", "AI fastest route")):
        pts = [pos[n] for n in path]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2.6, ls="--" if col == "black" else "-", label=lab)
    ax.scatter(*pos[src], c="lime", s=110, zorder=5, edgecolor="k")
    ax.scatter(*pos[dst], c="magenta", s=110, zorder=5, edgecolor="k")
    ax.legend(loc="upper left", fontsize=8); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or "Congestion-aware routing (edge colour = congestion)")
    return ax


if __name__ == "__main__":
    G = build_city_graph()
    apply_congestion(G, hotspot_congestion(G))
    m = evaluate_routing(G)
    (config.RESULTS_DIR / "routing_metrics.json").write_text(json.dumps(m, indent=2))
    print({k: round(v, 2) for k, v in m.items()})
    src, dst = (0, 3), (7, 4)
    a, b = shortest_distance_route(G, src, dst), fastest_route(G, src, dst)
    plot_routes(G, src, dst, a, b,
                title=f"Distance route {route_time(G, a):.1f} min vs AI route {route_time(G, b):.1f} min")
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "routing_demo.png", dpi=130)
