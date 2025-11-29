import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DirectedGraphIsing:
    """Ising model on a directed graph with Metropolis dynamics and animation."""

    def __init__(self, G, T=2.0, J=1.0):
        if not G.is_directed():
            raise ValueError("G must be a directed graph")
        self.G = G
        self.size = G.number_of_nodes()
        self.J = J
        self.beta = 1.0 / T
        # Initialize spins randomly
        self.spins = {node: np.random.choice([-1, 1]) for node in G.nodes}
        self.energy = self.compute_energy()
        self.magnetization = self.compute_magnetization()

    def compute_energy(self):
        """Compute energy for a directed graph: sum over all directed edges."""
        E = 0.0
        for i, j in self.G.edges:  # edge from i -> j
            E += -self.J * self.spins[i] * self.spins[j]
        return E

    def compute_magnetization(self):
        return sum(self.spins.values())

    def metropolis_step(self):
        """Perform a single Metropolis update considering incoming neighbors."""
        node = np.random.choice(list(self.G.nodes))
        s = self.spins[node]
        # Only consider incoming neighbors affecting this node
        neighbor_sum = sum(self.spins[nei] for nei in self.G.predecessors(node))
        delta_E = 2 * self.J * s * neighbor_sum
        if delta_E <= 0 or np.random.rand() < np.exp(-self.beta * delta_E):
            self.spins[node] *= -1
            self.energy += delta_E
            self.magnetization += 2 * self.spins[node]

    def run_animation(self, nt=200, interval=50):
        """Animate the Ising model on the directed graph with magnetization plot."""
        steps, magnet = [], []
        pos = nx.spring_layout(self.G, seed=42)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Initial node colors
        node_colors = ['red' if self.spins[n]==1 else 'black' for n in self.G.nodes]
        nodes = nx.draw_networkx_nodes(self.G, pos, node_color=node_colors,
                                       node_size=100, ax=ax1)
        nx.draw_networkx_edges(self.G, pos, ax=ax1, arrows=True)
        ax1.axis('off')
        ax1.set_title("Directed Graph Ising Spins")

        # Magnetization plot
        line, = ax2.plot([], [], lw=2)
        ax2.set_xlim(0, nt)
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel("MC cycles")
        ax2.set_ylabel("Magnetization")
        ax2.set_title("Magnetization vs MC cycles")

        def do_mc_cycle(n):
            for _ in range(self.size):
                self.metropolis_step()
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