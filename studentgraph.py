import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict

class StudentGraph:
    def __init__(self, fichier, associations_a_garder=None):
        """
        Initialise la classe avec le fichier CSV et les associations à garder.
        """
        self.fichier = fichier
        self.df = pd.read_csv(fichier)
        self.nom_colonne_eleve = self.df.columns[0]
        self.colonne_associations = "memberOf"
        self.df[self.colonne_associations] = self.df[self.colonne_associations].fillna("")
        self.df["liste_assos"] = self.df[self.colonne_associations].apply(
            lambda x: [a.strip() for a in x.split("|") if a.strip()]
        )

        if associations_a_garder is None:
            self.associations_a_garder = []
        else:
            self.associations_a_garder = associations_a_garder

        self.G = nx.Graph()
        self.couleurs_assos = {}
        self.node_colors = []
        self.node_hovertext = []

    def build_graph(self):
        """
        Construit le graphe NetworkX à partir du dataframe.
        """
        # Ajouter les nœuds
        for _, row in self.df.iterrows():
            self.G.add_node(row[self.nom_colonne_eleve])

        # Dictionnaire association -> membres
        asso_to_members = defaultdict(list)
        for _, row in self.df.iterrows():
            for asso in row["liste_assos"]:
                if asso in self.associations_a_garder:
                    asso_to_members[asso].append(row[self.nom_colonne_eleve])

        # Créer les liens uniquement pour les associations choisies
        for asso, membres in asso_to_members.items():
            for i in range(len(membres)):
                for j in range(i + 1, len(membres)):
                    self.G.add_edge(membres[i], membres[j], association=asso)

        # Supprimer les nœuds isolés
        isolated_nodes = list(nx.isolates(self.G))
        self.G.remove_nodes_from(isolated_nodes)

        # Préparer la palette de couleurs
        palette_couleurs = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
        ]
        self.couleurs_assos = {a: palette_couleurs[i % len(palette_couleurs)]
                               for i, a in enumerate(self.associations_a_garder)}

        # Déterminer la couleur et le hovertext de chaque nœud
        self.node_colors = []
        self.node_hovertext = []
        for node in self.G.nodes():
            assos = [a for a in self.df.loc[self.df[self.nom_colonne_eleve] == node, "liste_assos"].iloc[0]
                     if a in self.associations_a_garder]
            if len(assos) == 0:
                couleur = "#cccccc"  # gris
            elif len(assos) == 1:
                couleur = self.couleurs_assos[assos[0]]
            else:
                couleur = "black"  # plusieurs associations
            self.node_colors.append(couleur)
            self.node_hovertext.append(f"{node}<br>{' | '.join(assos)}")

    def get_graph(self):
        """
        Retourne le graphe NetworkX construit.
        """
        return self.G

    def plot_graph(self, title=None):
        """
        Affiche le graphe avec Plotly.
        """
        if self.G.number_of_nodes() == 0:
            print("Aucun nœud à afficher.")
            return

        pos = nx.spring_layout(self.G, seed=42)

        # Traces des arêtes
        edge_x, edge_y = [], []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Traces des nœuds
        node_x, node_y = [], []
        for node in self.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=self.node_hovertext,
            marker=dict(
                showscale=False,
                color=self.node_colors,
                size=12,
                line_width=2
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=title if title else f"Graphe des élèves (associations : {', '.join(self.associations_a_garder)})",
                    font=dict(size=16)
                ),
                showlegend=True,
                legend=dict(
                    title="Associations",
                    orientation="v",
                    x=1.02,
                    y=1,
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="black",
                    borderwidth=1
                ),
                hovermode='closest',
                margin=dict(b=0, l=0, r=150, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        # Ajouter la légende des couleurs
        for asso, couleur in self.couleurs_assos.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=couleur),
                name=asso
            ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='black'),
            name='Plusieurs associations'
        ))

        fig.show()