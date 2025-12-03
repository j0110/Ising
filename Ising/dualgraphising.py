import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DualGraphIsing:
    """
    Two-layer Ising model (A and B) on the same graph,
    with interlayer coupling C.
    """
    def __init__(self, G, T=2.0, J_A=1.0, J_B=1.0, C=0.2):
        self.G = G
        self.size = G.number_of_nodes()
        self.beta = 1.0 / T
        self.J_A = J_A
        self.J_B = J_B
        self.C = C
        self.dim = 1  # Not used but kept for consistency
        self.length_cycle = self.size  # Not used but kept for consistency
        self._reset_spin()

    def _reset_spin(self, to_value=None):
        """Réinitialise les spins des deux couches."""
        if to_value is not None:
            self.spins_A = {node: to_value for node in self.G.nodes}
            self.spins_B = {node: to_value for node in self.G.nodes}
        else:
            self.spins_A = {node: np.random.choice([-1, 1]) for node in self.G.nodes}
            self.spins_B = {node: np.random.choice([-1, 1]) for node in self.G.nodes}   

    def _get_energy(self):
        """Total energy of the two layers with interlayer coupling."""
        E_A, E_B, E_C = 0.0, 0.0, 0.0
        for i, j in self.G.edges:
            E_A += -self.J_A * self.spins_A[i] * self.spins_A[j]
            E_B += -self.J_B * self.spins_B[i] * self.spins_B[j]
        for node in self.G.nodes:
            E_C += -self.C * self.spins_A[node] * self.spins_B[node]
        return E_A + E_B + E_C

    def _get_magnetization(self, spins=None):
        """Normalized magnetization of a layer."""
        if spins is None:
            spins = self.spins_A
        return sum(spins.values()) / self.size

    def move(self):
        """Simple metropolis on both layers with inter-layer coupling."""
        for layer in ['A', 'B']:
            spins = self.spins_A if layer == 'A' else self.spins_B
            other = self.spins_B if layer == 'A' else self.spins_A
            J = self.J_A if layer == 'A' else self.J_B

            node = np.random.choice(list(self.G.nodes))
            s = spins[node]
            neighbor_sum = sum(spins[nei] for nei in self.G.neighbors(node))
            delta_E = 2 * s * (J * neighbor_sum + self.C * other[node])
            if delta_E <= 0 or np.random.rand() < np.exp(-self.beta * delta_E):
                spins[node] *= -1

    def make_animation(self, nt=200, frames_per_cycle=1, save_path="dual_ising.gif", interval=100):
        """
        Generates and saves a GIF animation of the model.
        - nt: number of frames
        - frames_per_cycle: how many Monte Carlo steps per frame
        - gif_path: name of the GIF file to save
        """
        pos = nx.spring_layout(self.G, seed=42)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axA, axB, axM = axes
        axA.axis('off'); axB.axis('off')
        axM.set_xlabel("Frames")
        axM.set_ylabel("Magnetisation")
        axM.set_ylim(-1, 1)
        lineA, = axM.plot([], [], 'r-', label='A')
        lineB, = axM.plot([], [], 'b-', label='B')
        axM.legend(loc='upper right')

        steps, magsA, magsB = [], [], []

        def update(frame):
            for _ in range(frames_per_cycle * self.size):
                self.move()

            axA.clear(); axB.clear(); axM.clear()
            axA.axis('off'); axB.axis('off')
            axM.set_xlabel("Frames"); axM.set_ylabel("Magnetisation")
            axM.set_ylim(-1, 1)

            # Layer A
            nx.draw_networkx_edges(self.G, pos, ax=axA)
            nx.draw_networkx_nodes(
                self.G, pos,
                node_color=['red' if self.spins_A[n]==1 else 'black' for n in self.G.nodes],
                node_size=120, ax=axA)
            axA.set_title("Decision A")

            # Layer B
            nx.draw_networkx_edges(self.G, pos, ax=axB)
            nx.draw_networkx_nodes(
                self.G, pos,
                node_color=['blue' if self.spins_B[n]==1 else 'black' for n in self.G.nodes],
                node_size=120, ax=axB)
            axB.set_title("Decision B")

            # Magnetisations
            mA = self._get_magnetization(self.spins_A)
            mB = self._get_magnetization(self.spins_B)
            magsA.append(mA); magsB.append(mB); steps.append(frame)
            axM.plot(steps, magsA, 'r-')
            axM.plot(steps, magsB, 'b-')
            axM.set_title("Magnetisations (A red, B blue)")
            axM.set_xlim(0, nt)
            return []

        anim = animation.FuncAnimation(fig, update, frames=nt, interval=interval, blit=False)
        writer = animation.PillowWriter(fps=1000//interval)
        anim.save(save_path, writer=writer)
