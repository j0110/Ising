import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

class NormalIsing:
    def __init__(self, T, J, L, dim, h=0, mode="normal", epsilon=0.5, wolff=False):
        self.dim = dim
        self.size = L
        self.J = J
        self.beta = 1./T
        self.h = h
        self._reset_spin()
        self.energy = self._get_energy()
        self.magnetization = self._get_magnetization()
        self.mode = mode
        self.epsilon = epsilon
        if not wolff:
            self.move = self.metropolis_move
            self.length_cycle = self.size ** self.dim
        else:
            self.move = self.wolff_move
            self.length_cycle = 1

    def _reset_spin(self, to_value=None):
        """Réinitialise les spins."""
        if to_value is not None:
            self.spins = to_value*np.ones(shape=tuple([self.size]*self.dim))
        else:
            self.spins = np.random.choice([-1,1], size=tuple([self.size]*self.dim))
    
    def _get_neighbors(self, idx, only_forward=False):
        neighbors = []
        for d in range(self.dim):
            # Forward neighbor along dimension d
            fwd = list(idx)
            fwd[d] = (fwd[d] + 1) % self.size
            neighbors.append(self.spins[tuple(fwd)])
            if not only_forward:
                # Backward neighbor along dimension d
                bwd = list(idx)
                bwd[d] = (bwd[d] - 1) % self.size
                neighbors.append(self.spins[tuple(bwd)])
        return neighbors

    def _get_energy(self):
        """Compute the total energy for 2D or 3D Ising configuration."""
        energ = 0.0
        for idx in itertools.product(range(self.size), repeat=self.dim):
                # Only count "forward" neighbors to avoid double counting
                energ += -self.J * self.spins[idx] * (np.sum(self._get_neighbors(idx, only_forward = True)))
        energ += - self.h * np.sum(self.spins)
        return energ

    def _get_magnetization(self):
        """Returns the total magnetization"""
        return np.sum(self.spins)

    def metropolis_move(self):
        idx = tuple(np.random.randint(self.size, size=self.dim))
        spin = self.spins[idx]
        neighbors = self._get_neighbors(idx)
        total_neighbor = sum(neighbors)
        if self.mode == 'normal':
            # Ising classique
            delta_energy = 2 * self.J * spin * total_neighbor + 2 * self.h * spin
            prob_flip = np.exp(-self.beta * delta_energy)
        elif self.mode == 'self_identity':
            if spin * total_neighbor >= 0:  # majorité comme lui
                # Ising classique
                delta_energy = 2 * self.J * spin * total_neighbor + 2 * self.h * spin
                prob_flip = np.exp(-self.beta * delta_energy)
            else:  # majorité différente
                delta_energy = 0  # énergie ignorée dans ce cas
                prob_flip = 1 - self.epsilon
        if np.random.random() < prob_flip:
            self.spins[idx] *= -1
            self.energy += delta_energy
            self.magnetization += 2 * self.spins[idx]

    def wolff_move(self):
        p = 1.0 - np.exp(-2.0 * self.beta)
        L = self.size
        dim = self.dim
        start_idx = tuple(np.random.randint(L, size=dim))
        target_spin = self.spins[start_idx]
        in_cluster = np.zeros_like(self.spins, dtype=bool)
        in_cluster[start_idx] = True
        stack = [start_idx]
        directions = []
        for d in range(dim):
            e = [0] * dim
            e[d] = 1
            directions.append(tuple(e))
            e = [0] * dim
            e[d] = -1
            directions.append(tuple(e))
        while stack:
            idx = stack.pop()
            for d in directions:
                neigh_idx = tuple((idx[i] + d[i]) % L for i in range(dim))
                if (not in_cluster[neigh_idx]) and (self.spins[neigh_idx] == target_spin):
                    if np.random.random() < p:
                        in_cluster[neigh_idx] = True
                        stack.append(neigh_idx)
        self.spins[in_cluster] *= -1
        self.energy = self._get_energy()
        self.magnetization = self._get_magnetization()

    def _get_plot_data(self):
        """Return positions and colors for scatter"""
        coords = np.array(list(itertools.product(range(self.size), repeat=self.dim)), dtype=np.float32)
        colors = np.array(['red' if s==1 else 'black' for s in self.spins.flatten()])
        return coords, colors

    def run_animation(self, nt=200, interval=100, save_path="ising_animation.gif"):
        """Run animation using matplotlib for 2D and 3D Ising configurations with sliding magnetization"""
        steps, magnet = [], []
        fig = plt.figure()
        if self.dim == 3:
            ax = fig.add_subplot(121, projection='3d')
            coords, colors = self._get_plot_data()
            plotter = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=colors, s=100)
            ax.set_xlim(-1, self.size)
            ax.set_ylim(-1, self.size)
            ax.set_zlim(-1, self.size)
            ax.set_title("3D Spin configuration")
        else:  # 2D
            ax = fig.add_subplot(121)
            plotter = ax.imshow(self.spins, interpolation='none',vmin=-1,vmax=1, cmap=ListedColormap(['red','black']))
            ax.set_xlim(-1, self.size)
            ax.set_ylim(-1, self.size)
            ax.set_title("2D Spin configuration")
        ax2 = fig.add_subplot(122)
        line, = ax2.plot([], [])
        ax2.set_xlim(0, nt)
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel("MC cycles")
        ax2.set_ylabel("Magnetization")

        def do_mc_cycle(frame):
            # Perform one MC sweep per frame
            for _ in range(self.length_cycle):
                self.move()
            coords, colors = self._get_plot_data()
            if self.dim == 3:
                plotter._offsets3d = (coords[:,0], coords[:,1], coords[:,2])
                plotter.set_color(colors)
            else:
                plotter.set_data(self.spins)
            m = self.magnetization / float(self.size ** self.dim)
            if len(steps) < nt:
                steps.append(frame)
            if len(magnet) < nt:
                magnet.append(m)
            else:
                magnet.insert(nt, m)
                magnet.pop(0)
            line.set_data(steps, magnet)
            return plotter, line

        self.anim = animation.FuncAnimation(fig, do_mc_cycle, frames=nt, interval=interval, blit=False)
        writer = animation.PillowWriter(fps=1000//interval)
        self.anim.save(save_path, writer=writer)