import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DualGraphIsing:
    """
    Modèle d'Ising à deux couches (A et B) sur le même graphe,
    avec un couplage inter-couche C.
    """
    def __init__(self, G, T=2.0, J_A=1.0, J_B=1.0, C=0.2):
        self.G = G
        self.size = G.number_of_nodes()
        self.beta = 1.0 / T
        self.J_A = J_A
        self.J_B = J_B
        self.C = C

        # Spins initiaux aléatoires pour chaque couche
        self.spins_A = {node: np.random.choice([-1, 1]) for node in G.nodes}
        self.spins_B = {node: np.random.choice([-1, 1]) for node in G.nodes}

    def compute_magnetization(self, spins):
        """Magnétisation normalisée d'une couche."""
        return sum(spins.values()) / self.size

    def metropolis_step(self):
        """Metropolis simple sur les deux couches avec couplage inter-couche."""
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

    def make_animation(self, nt=200, frames_per_cycle=1, gif_path="dual_ising.gif",
                       interval=100, dpi=100):
        """
        Génère et sauvegarde une animation GIF du modèle.
        - nt : nombre de frames
        - frames_per_cycle : combien de pas Monte Carlo par frame
        - gif_path : nom du fichier GIF à sauvegarder
        """
        pos = nx.spring_layout(self.G, seed=42)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axA, axB, axM = axes
        axA.axis('off'); axB.axis('off')
        axM.set_xlabel("Frames")
        axM.set_ylabel("Magnétisation")
        axM.set_ylim(-1, 1)
        lineA, = axM.plot([], [], 'r-', label='A')
        lineB, = axM.plot([], [], 'b-', label='B')
        axM.legend(loc='upper right')

        steps, magsA, magsB = [], [], []

        def update(frame):
            for _ in range(frames_per_cycle * self.size):
                self.metropolis_step()

            axA.clear(); axB.clear(); axM.clear()
            axA.axis('off'); axB.axis('off')
            axM.set_xlabel("Frames"); axM.set_ylabel("Magnétisation")
            axM.set_ylim(-1, 1)

            # Couche A
            nx.draw_networkx_edges(self.G, pos, ax=axA)
            nx.draw_networkx_nodes(
                self.G, pos,
                node_color=['red' if self.spins_A[n]==1 else 'black' for n in self.G.nodes],
                node_size=120, ax=axA)
            axA.set_title("Décision A")

            # Couche B
            nx.draw_networkx_edges(self.G, pos, ax=axB)
            nx.draw_networkx_nodes(
                self.G, pos,
                node_color=['blue' if self.spins_B[n]==1 else 'black' for n in self.G.nodes],
                node_size=120, ax=axB)
            axB.set_title("Décision B")

            # Magnétisations
            mA = self.compute_magnetization(self.spins_A)
            mB = self.compute_magnetization(self.spins_B)
            magsA.append(mA); magsB.append(mB); steps.append(frame)
            axM.plot(steps, magsA, 'r-')
            axM.plot(steps, magsB, 'b-')
            axM.set_title("Magnétisations (A rouge, B bleu)")
            axM.set_xlim(0, nt)
            return []

        ani = animation.FuncAnimation(fig, update, frames=nt,
                                      interval=interval, blit=False)

        # Sauvegarde en GIF (avec PillowWriter)
        writer = animation.PillowWriter(fps=1000/interval)
        ani.save(gif_path, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Animation sauvegardée sous : {gif_path}")