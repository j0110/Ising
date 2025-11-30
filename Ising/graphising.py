import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .utils import get_members_of_association

class GraphIsing:
    """Ising model on an arbitrary graph with working animation."""

    def __init__(self, G, T=2.0, J=1.0, influent_association=None, student_graph=None):
        self.G = G
        self.size = G.number_of_nodes()
        self.dim = 1
        self.length_cycle = self.size  # one MC cycle = N updates
        self.J = J
        self.beta = 1.0 / T
        self.spins = {node: np.random.choice([-1, 1]) for node in G.nodes}
        if influent_association:
            influencer_nodes = get_members_of_association(student_graph, influent_association) if influent_association else None
            self.influencer_nodes = set(influencer_nodes) if influencer_nodes else set()  # nœuds bloqués
            self.spins = {node: (1 if node in influencer_nodes else -1) for node in G.nodes}
        self.energy = self._get_energy()
        self.magnetization = self._get_magnetization()


    def _get_energy(self):
        E = 0.0
        for i, j in self.G.edges:
            E += -self.J * self.spins[i] * self.spins[j]
        return E

    def _get_magnetization(self):
        return sum(self.spins.values())

    def move(self):
        node = np.random.choice(list(self.G.nodes))
        if node in self.influencer_nodes:
            return
        s = self.spins[node]
        neighbor_sum = sum(self.spins[nei] for nei in self.G.neighbors(node))
        delta_E = 2 * self.J * s * neighbor_sum
        if delta_E <= 0 or np.random.rand() < np.exp(-self.beta * delta_E):
            self.spins[node] *= -1
            self.energy += delta_E
            self.magnetization += 2 * self.spins[node]

    def run_animation(self, nt=200, interval=50):
        """Animate the Ising model with magnetization plot like in 2D case."""
        steps, magnet = [], []
        pos = nx.spring_layout(self.G, seed=42)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Draw initial graph with red for +1, black for -1
        node_colors = ['red' if self.spins[n]==1 else 'black' for n in self.G.nodes]
        nodes = nx.draw_networkx_nodes(self.G, pos, node_color=node_colors,
                                       node_size=100, ax=ax1)
        nx.draw_networkx_edges(self.G, pos, ax=ax1)
        ax1.axis('off')
        ax1.set_title("Graph Ising Spins")

        # Setup magnetization plot
        line, = ax2.plot([], [], lw=2)
        ax2.set_xlim(0, nt)
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel("MC cycles")
        ax2.set_ylabel("Magnetization")
        ax2.set_title("Magnetization vs MC cycles")

        def do_mc_cycle(n):
            # Perform N Metropolis steps per cycle
            for _ in range(self.size):
                self.move()

            # Update node colors
            new_colors = ['red' if self.spins[n]==1 else 'black' for n in self.G.nodes]
            nodes.set_facecolor(new_colors)

            # Update magnetization
            m = self.magnetization / self.size
            if len(steps) < nt:
                steps.append(n)
                magnet.append(m)
            else:
                magnet.pop(0)
                magnet.append(m)
            line.set_data(range(len(magnet)), magnet)
            return nodes, line

        self.anim = animation.FuncAnimation(fig, do_mc_cycle, frames=nt,
                                            interval=interval, blit=False, cache_frame_data=False)
        plt.show()