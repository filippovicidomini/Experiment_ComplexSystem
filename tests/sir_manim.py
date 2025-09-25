
from manim import *
import networkx as nx
import random

# -------------------------
# SIR dynamics (pure NetworkX)
# -------------------------
# States: 0=S (susceptible), 1=I (infected), 2=R (recovered)
def sir_steps(G, beta=0.08, gamma=0.05, initial_infected=None, steps=80, seed=7):
    random.seed(seed)
    state = {u:0 for u in G.nodes()}
    if initial_infected is None:
        initial_infected = max(G.degree, key=lambda x: x[1])[0]
    state[initial_infected] = 1

    history = [state.copy()]
    for t in range(steps):
        # stop if no infected remain
        if not any(v==1 for v in state.values()):
            break
        new_state = state.copy()
        # infections
        for u in G.nodes():
            if state[u] == 1:
                for v in G.neighbors(u):
                    if state[v] == 0 and random.random() < beta:
                        new_state[v] = 1
        # recoveries
        for u in G.nodes():
            if state[u] == 1 and random.random() < gamma:
                new_state[u] = 2
        state = new_state
        history.append(state.copy())
    return history

# -------------------------
# Manim Scene
# -------------------------
class SIRKarateAnimation(MovingCameraScene):
    def construct(self):
        # Parameters
        beta = 0.1     # infection probability per contact per step
        gamma = 0.05    # recovery probability per step
        steps = 60
        seed = 1

        # Build graph
        G = nx.karate_club_graph()
        # fai in modo che occupi tutta la scena orizzontalmente
        pos = nx.circular_layout(G)

        # Generate SIR state history
        history = sir_steps(G, beta=beta, gamma=gamma, steps=steps, seed=seed)

        # Colors for S, I, R
        STATE_COLOR = {0: BLUE_E, 1: YELLOW_E, 2: GREEN_E}

        # Build Manim Graph from NetworkX positions
        vertices = list(G.nodes())
        edges = list(G.edges())

        # convert positions to manim format
        layout = {v: np.array([pos[v][0], pos[v][1], 0.0]) * 4.0 for v in vertices}  # scale for visibility

        # Initial vertex config
        vconf = {v: {"fill_color": STATE_COLOR[history[0][v]], "stroke_width": 1, "radius": 0.15} for v in vertices}

        graph = Graph(vertices, edges, layout=layout, vertex_config=vconf, edge_config={"stroke_opacity": 0.6})
        title = Text("SIR on Karate Club", weight=BOLD).scale(0.8).to_edge(UP)

        # Legend
        box = VGroup(
            Dot(color=BLUE_E).scale(1.2), Text("S", font_size=28),
            Dot(color=YELLOW_E).scale(1.2), Text("I", font_size=28),
            Dot(color=GREEN_E).scale(1.2), Text("R", font_size=28),
        ).arrange(RIGHT, buff=0.4)
        legend = VGroup(box).to_edge(DOWN)

        # Counter
        step_counter = Text("t = 0", font_size=28).next_to(title, DOWN).shift(0.2*DOWN)

        # Camera framing
        self.camera.frame.save_state()
        self.play(FadeIn(title), FadeIn(step_counter))
        self.play(Create(graph), FadeIn(legend))
        self.wait(0.5)

        # Animate transitions
        for t in range(1, len(history)):
            new_state = history[t]
            changes = []
            for v in vertices:
                old_c = graph[v].get_fill_color()
                new_c = STATE_COLOR[new_state[v]]
                if old_c.to_hex() != new_c.to_hex():
                    changes.append(graph[v].animate.set_fill(new_c))
            # Update the step counter
            new_counter = Text(f"t = {t}", font_size=28).next_to(title, DOWN).shift(0.2*DOWN)
            self.play(Transform(step_counter, new_counter), *changes, run_time=0.6)
            self.wait(0.1)

        self.wait(1.5)

# ---------------
# Tips to run:
# manim -pql sir_manim.py SIRKarateAnimation
# (for higher quality use -pqh or -pqk)
# Ensure you have Manim Community installed: pip install manim
# Also requires: networkx
