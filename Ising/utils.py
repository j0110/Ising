import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import Image, display
from tqdm import tqdm
from scipy.optimize import curve_fit

def compute_properties(model, var_name, var_value, n_warmup=1000, n_cycles=100, n_average = 1, reset_state=True):
    """
    Compute <M>, <E>, χ, and C vs T or h using a Monte Carlo simulation
    with warm-up and measurement cycles, showing progress with tqdm.
    """
    results = {var_name: var_value, 'M': [], 'E': [], 'chi': [], 'C': []}
    N = model.size ** model.dim  # total number of spins

    if not reset_state:
        n_average = 1  # Disable averaging if not resetting state

    for var in tqdm(results[var_name], desc="Computing  properties"):
        if var_name == 'T':
            model.beta = 1. / var
        elif var_name == 'h':
            model.h = var
        else:
            raise ValueError("var_name must be 'T' or 'h'")
        av_m, av_m2, av_e, av_e2 = 0, 0, 0, 0
        
        for _ in range(n_average):
            # --- Warm-up phase ---
            if reset_state:
                model._reset_spin()
                model.energy = model._get_energy()
                model.magnetization = model._get_magnetization()

            for _ in range(n_warmup * model.length_cycle):
                model.move()

            # --- Measurement phase ---
            
            for _ in range(n_cycles):
                # Perform one MC sweep (N updates)
                for _ in range(model.length_cycle):
                    model.move()

                # Accumulate averages
                m = model._get_magnetization()
                e = model._get_energy()
                av_m += m
                av_m2 += m**2
                av_e += e
                av_e2 += e**2

        # Normalize averages
        av_m /= n_cycles * n_average
        av_m2 /= n_cycles * n_average
        av_e /= n_cycles * n_average
        av_e2 /= n_cycles * n_average

        fact = 1.0 / N
        T = 1. / model.beta
        k_B = 1.0

        # Store results
        if var_name=="h":
            results['M'].append(fact * av_m)
        elif var_name=="T":
            results['M'].append(fact * np.abs(av_m))
        results['E'].append(fact * av_e)
        results['C'].append((fact * (av_e2 - av_e**2) / (k_B * T**2)))
        results['chi'].append(fact * (av_m2 - av_m**2) / (k_B * T))

    return results


def plot_properties(var_name, results, save_path="thermal_properties.png"):
    """Plot <M>, χ, and C vs T and save as an image."""
    x_axis = results[var_name]

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, [elem/np.max(np.abs(results['M'])) for elem in results['M']], '+-b', label='<|M|>')
    plt.plot(x_axis, [elem/np.max(np.abs(results['E'])) for elem in results['E']], '+-m', label='<E>')
    plt.plot(x_axis, [elem/np.max(np.abs(results['chi'])) for elem in results['chi']], '+-g', label='χ (susceptibility)')
    plt.plot(x_axis, [elem/np.max(np.abs(results['C'])) for elem in results['C']], '+-r', label='C (specific heat)')

    plt.xlabel('Temperature (T)' if var_name == 'T' else 'Magnetic field (h)')
    plt.ylabel('Arbitrary scale')
    plt.title('Thermal Properties vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    plt.savefig(save_path)
    plt.close()  # Close the figure to avoid showing it inline

    # Display in notebook
    display(Image(filename=save_path))


def compute_critical_exponents(results, Tc_guess):
    """
    Fit critical exponents β, γ, α from Monte Carlo results near a given Tc.

    Parameters
    ----------
    results : dict
        Dictionary with keys 'T', 'M', 'chi', 'C' from compute_thermal_properties or compute_thermal_properties_average.
    Tc_guess : float
        Initial estimate of the critical temperature.

    Returns
    -------
    dict
        Dictionary containing:
        - Fitted critical temperatures: 'Tc_M', 'Tc_chi', 'Tc_C'
        - Critical exponents: 'beta', 'gamma', 'alpha'
        - Amplitudes: 'A_M', 'A_chi', 'A_C'
    """

    T = np.array(results['T'])
    M = np.array(results['M'])
    chi = np.array(results['chi'])
    C = np.array(results['C'])

    # --- Define power-law functions for fitting ---
    def M_law(T, Tc, beta, A):
        return A * np.abs(Tc - T)**beta

    def chi_law(T, Tc, gamma, A):
        return A * np.abs(T - Tc)**(-gamma)

    def C_law(T, Tc, alpha, A):
        return A * np.abs(T - Tc)**(-alpha)

    # --- Fit magnetization for T < Tc ---
    mask_M = T < Tc_guess
    popt_M, _ = curve_fit(M_law, T[mask_M], M[mask_M],
                          p0=[Tc_guess, 0.125, 1.0], maxfev=5000)
    Tc_fit_M, beta_fit, A_M = popt_M

    # --- Fit susceptibility around Tc ---
    mask_chi = (T > Tc_guess - 0.5) & (T < Tc_guess + 0.5)
    popt_chi, _ = curve_fit(chi_law, T[mask_chi], chi[mask_chi],
                            p0=[Tc_guess, 1.75, 1.0], maxfev=5000)
    Tc_fit_chi, gamma_fit, A_chi = popt_chi

    # --- Fit specific heat around Tc ---
    mask_C = (T > Tc_guess - 0.5) & (T < Tc_guess + 0.5)
    popt_C, _ = curve_fit(C_law, T[mask_C], C[mask_C],
                           p0=[Tc_guess, 0.0, 1.0], maxfev=5000)
    Tc_fit_C, alpha_fit, A_C = popt_C

    # --- Combine results into a dictionary ---
    results_exponents = {
        'Tc_M': Tc_fit_M, 'beta': beta_fit, 'A_M': A_M,
        'Tc_chi': Tc_fit_chi, 'gamma': gamma_fit, 'A_chi': A_chi,
        'Tc_C': Tc_fit_C, 'alpha': alpha_fit, 'A_C': A_C
    }

    return results_exponents


def get_members_of_association(studentgraph, association):
    """Retourne la liste des membres d'une association donnée."""
    return [
        row[studentgraph.nom_colonne_eleve]
        for _, row in studentgraph.df.iterrows()
        if association in row["liste_assos"]
    ]

def iterations_to_treshold(class_model, var_name, var_values, kargs, iter_per_value, max_step, treshold):
    results = {}
    for var in tqdm(var_values, desc=f"Progress over {var_name}"):
        results[var] = []
        all_args = {**{var_name: var}, **kargs}
        for _ in tqdm(range(iter_per_value), desc=f"  Iterations for {var_name}={var}", leave=False):
            model = class_model(**all_args)
            threshold_reached = False
            inv_size = 1.0 / model.size
            for step in range(max_step):
                model.move()
                m = model.magnetization * inv_size
                if m > treshold:
                    threshold_reached = True
                    break
            if threshold_reached:
                results[var].append(step)
    return results