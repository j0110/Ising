A Python library implementing various Ising model simulations using **Monte Carlo methods**, including the **Metropolis** and **Wolff cluster algorithms** (used for efficient simulation near critical points).

The core analysis covers four main areas:
* **Self-Identity:** Incorporates a parameter ($\epsilon$) to model individual adherence to personal choice versus conformity with the social environment.
* **External Pressure ($H$):** Studies the effects of a global bias (analogous to external pressure) on the system, observing **nucleation points** and **hysteresis** phenomena.
* **Geometrical Effects:** Extends the model to 3D lattices and complex topologies, including **non-oriented, oriented graphs** and a **multi-component (dual-layer) coupled Ising model**.
* **Real Network Application:** Maps the model onto a **real student interaction network** to analyze **rumor propagation** and find physical properties like critical temperature and exponents for the specific graph structure.


____________________________________________________________________________________________________________________________________________
**Quick Start: How to Use Our Code**

Our entire analysis is packaged into a single, executable notebook.

1.  **Clone the Repository (optional):**
    ```bash
    git clone [https://github.com/j0110/Ising.git](https://github.com/j0110/Ising.git)
    cd Ising
    ```
2.  **Run the Analysis:**
    * Open the file `Ising.ipynb` in **Jupyter Notebook** or **VS Code**.
    * Select **Run All Cells**.

> **Note:** The notebook is designed to call functions and load data from the repository's files, making the execution of the entire analysis fast and self-contained.
