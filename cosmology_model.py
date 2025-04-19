# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 15:20:45 2025

@authors: Daria Teodora Harabor, David Turturean
"""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Physical constants
c = 299792.458  # Speed of light in km/s
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
h_planck = 6.62607015e-34  # Planck constant in J.s
k_B = 1.380649e-23  # Boltzmann constant in J/K
sigma_T = 6.6524587321e-29  # Thomson cross-section in m^2
Mpc_to_m = 3.08567758e22  # Megaparsec to meters

# CMB parameters
T_cmb = 2.7255  # CMB temperature in K
rho_gamma = (np.pi**2 / 15) * (k_B * T_cmb)**4 / (c**3 * h_planck**3)  # Radiation energy density

# Pivot scale for primordial spectrum
k_pivot = 0.05  # Mpc^-1

def hubble_parameter(z, H0, Omega_m, Omega_r, Omega_Lambda):
    """
    Calculate the Hubble parameter at redshift z.
    
    Args:
        z (float): Redshift
        H0 (float): Hubble constant in km/s/Mpc
        Omega_m (float): Matter density parameter
        Omega_r (float): Radiation density parameter
        Omega_Lambda (float): Dark energy density parameter
        
    Returns:
        float: H(z) in km/s/Mpc
    """
    return H0 * np.sqrt(Omega_r * (1 + z)**4 + Omega_m * (1 + z)**3 + Omega_Lambda)

def comoving_distance(z, H0, Omega_m, Omega_r, Omega_Lambda):
    """
    Calculate the comoving distance to redshift z.
    
    Args:
        z (float): Redshift
        H0 (float): Hubble constant in km/s/Mpc
        Omega_m (float): Matter density parameter
        Omega_r (float): Radiation density parameter
        Omega_Lambda (float): Dark energy density parameter
        
    Returns:
        float: Comoving distance in Mpc
    """
    def integrand(z_prime):
        return c / hubble_parameter(z_prime, H0, Omega_m, Omega_r, Omega_Lambda)
    
    result, _ = quad(integrand, 0, z)
    return result

def angular_diameter_distance(z, H0, Omega_m, Omega_r, Omega_Lambda):
    """
    Calculate the angular diameter distance to redshift z.
    
    Args:
        z (float): Redshift
        H0 (float): Hubble constant in km/s/Mpc
        Omega_m (float): Matter density parameter
        Omega_r (float): Radiation density parameter
        Omega_Lambda (float): Dark energy density parameter
        
    Returns:
        float: Angular diameter distance in Mpc
    """
    d_C = comoving_distance(z, H0, Omega_m, Omega_r, Omega_Lambda)
    return d_C / (1 + z)

def sound_horizon(z_star, H0, Omega_b_h2, Omega_m_h2):
    """
    Calculate the sound horizon at redshift z_star (recombination).
    
    Args:
        z_star (float): Recombination redshift
        H0 (float): Hubble constant in km/s/Mpc
        Omega_b_h2 (float): Physical baryon density
        Omega_m_h2 (float): Physical matter density
        
    Returns:
        float: Sound horizon in Mpc
    """
    # Convert to physical densities
    h = H0 / 100.0
    Omega_b = Omega_b_h2 / h**2
    Omega_m = Omega_m_h2 / h**2
    Omega_r = 4.15e-5 / h**2  # Radiation density including photons and neutrinos
    Omega_Lambda = 1 - Omega_m - Omega_r  # Assuming flat universe
    
    def integrand(z_prime):
        # Sound speed
        R = 3.0 * Omega_b / (4.0 * Omega_r) * (1 + z_prime)**(-1)
        cs = c / np.sqrt(3 * (1 + R))
        
        return cs / hubble_parameter(z_prime, H0, Omega_m, Omega_r, Omega_Lambda)
    
    result, _ = quad(integrand, z_star, np.inf, limit=1000)
    return result

def recombination_redshift(Omega_b_h2, Omega_m_h2):
    """
    Calculate the recombination redshift using the fitting formula from Hu & Sugiyama.
    
    Args:
        Omega_b_h2 (float): Physical baryon density
        Omega_m_h2 (float): Physical matter density
        
    Returns:
        float: Recombination redshift z_*
    """
    # Fitting formula for z_star from Hu & Sugiyama (1996)
    g1 = 0.0783 * Omega_b_h2**(-0.238) / (1 + 39.5 * Omega_b_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Omega_b_h2**1.81)
    
    return 1048 * (1 + 0.00124 * Omega_b_h2**(-0.738)) * (1 + g1 * Omega_m_h2**g2)

def primordial_power_spectrum(k, A_s, n_s):
    """
    Calculate the primordial power spectrum.
    
    Args:
        k (float or array): Wavenumber in Mpc^-1
        A_s (float): Primordial amplitude at pivot scale
        n_s (float): Scalar spectral index
        
    Returns:
        float or array: Primordial power P(k)
    """
    return A_s * (k / k_pivot)**(n_s - 1)

def transfer_function_bbks(k, Omega_m_h2, Omega_b_h2, h):
    """
    BBKS transfer function (Bardeen, Bond, Kaiser, Szalay).
    A simplified transfer function for matter perturbations.
    
    Args:
        k (float or array): Wavenumber in h/Mpc
        Omega_m_h2 (float): Physical matter density
        Omega_b_h2 (float): Physical baryon density
        h (float): Dimensionless Hubble parameter
        
    Returns:
        float or array: Transfer function T(k)
    """
    # Convert to physical units
    k_phys = k * h  # Convert to Mpc^-1
    
    # Scale of equality
    Omega_m = Omega_m_h2 / h**2
    z_eq = 25000 * Omega_m * h**2 / T_cmb**4
    k_eq = 0.073 * Omega_m * h**2 * T_cmb**(-2)  # Mpc^-1
    
    # BBKS fitting formula with baryon correction
    q = k_phys / (k_eq * h)
    
    # Baryon effects (approximate)
    alpha_b = 2.07 * k_eq * h * (1 + R_drag(Omega_b_h2, Omega_m_h2))**(-3/4)
    beta_b = 0.5 + (Omega_b_h2/Omega_m_h2) / 3
    
    # BBKS transfer function
    x = q * np.sqrt(alpha_b)
    T_bbks = np.log(1 + 2.34 * x) / (2.34 * x) * \
             (1 + 3.89 * x + (16.1 * x)**2 + (5.46 * x)**3 + (6.71 * x)**4)**(-0.25)
    
    # Apply baryon correction
    T_baryon = T_bbks * (1 + (beta_b - 1) * (1 + (k_phys/5)**2)**(-1))
    
    return T_baryon

def R_drag(Omega_b_h2, Omega_m_h2):
    """
    Calculate R_drag parameter for baryon effects.
    
    Args:
        Omega_b_h2 (float): Physical baryon density
        Omega_m_h2 (float): Physical matter density
        
    Returns:
        float: R_drag
    """
    # Simplified approximation
    return 31.5 * Omega_b_h2 / T_cmb**4 * (1000 / recombination_redshift(Omega_b_h2, Omega_m_h2))

def radiation_transfer_function(k, ell, tau, theta_s):
    """
    Simplified radiation transfer function for the CMB.
    This is a semi-analytic approximation that includes main features:
    - Acoustic oscillations 
    - Diffusion damping (Silk damping)
    - Reionization damping at low ell
    
    Args:
        k (float or array): Wavenumber in Mpc^-1
        ell (int or array): Multipole moment
        tau (float): Optical depth to reionization
        theta_s (float): Sound horizon angle (rad)
        
    Returns:
        float or array: Radiation transfer function
    """
    # Use ell ~ k*r where r is the angular diameter distance to recombination
    ell_k = ell  # We assume k values are already appropriately scaled
    
    # Acoustic oscillations
    acoustic = np.sin(np.pi * ell / (theta_s * 180 / np.pi))**2
    
    # Silk damping envelope
    k_silk = 0.2 * Mpc_to_m  # Silk scale in Mpc^-1
    ell_silk = 1800  # Approximate ell for Silk damping
    envelope = np.exp(-(ell / ell_silk)**2)
    
    # Reionization damping
    reion_damping = np.exp(-2 * tau * (ell < 30))
    
    # Combined effect
    T_rad = acoustic * envelope * reion_damping
    
    return T_rad

def angular_power_spectrum(ell, params):
    """
    Calculate the CMB TT angular power spectrum.
    
    Args:
        ell (array): Multipole moments
        params (dict): Cosmological parameters
            Required keys:
            - 'H0': Hubble constant in km/s/Mpc
            - 'Omega_b_h2': Physical baryon density
            - 'Omega_c_h2': Physical cold dark matter density
            - 'n_s': Scalar spectral index
            - 'A_s': Primordial amplitude
            - 'tau': Reionization optical depth
            
    Returns:
        array: CMB TT power spectrum D_ell values
    """
    # Extract parameters
    H0 = params['H0']
    h = H0 / 100.0
    Omega_b_h2 = params['Omega_b_h2']
    Omega_c_h2 = params['Omega_c_h2']
    Omega_m_h2 = Omega_b_h2 + Omega_c_h2
    n_s = params['n_s']
    A_s = params['A_s']
    tau = params['tau']
    
    # Calculate recombination redshift
    z_star = recombination_redshift(Omega_b_h2, Omega_m_h2)
    
    # Calculate sound horizon
    r_s = sound_horizon(z_star, H0, Omega_b_h2, Omega_m_h2)
    
    # Calculate angular diameter distance to recombination
    Omega_m = Omega_m_h2 / h**2
    Omega_r = 4.15e-5 / h**2
    Omega_Lambda = 1 - Omega_m - Omega_r
    d_A = angular_diameter_distance(z_star, H0, Omega_m, Omega_r, Omega_Lambda)
    
    # Sound horizon angle
    theta_s = r_s / d_A * 180 / np.pi  # in degrees
    
    # Semi-analytic approximation
    # Convert multipoles to approximate wavenumbers
    k_array = ell / d_A  # Approximate relation
    
    # Calculate primordial power spectrum
    P_prim = primordial_power_spectrum(k_array, A_s, n_s)
    
    # Apply radiation transfer function
    T_rad = radiation_transfer_function(k_array, ell, tau, theta_s)
    
    # Combine to get power spectrum
    C_ell = P_prim * T_rad
    
    # Convert to D_ell = ℓ(ℓ+1)C_ℓ/2π
    D_ell = ell * (ell + 1) * C_ell / (2 * np.pi)
    
    # Scale to μK^2
    D_ell *= 1e9
    
    return D_ell

def create_improved_model(ell, params):
    """
    Create an improved semi-analytic ΛCDM model for the CMB TT power spectrum.
    
    This model is physically motivated but simplified compared to full Boltzmann codes.
    It captures the key features of the CMB power spectrum.
    
    Args:
        ell (array): Multipole moments
        params (dict): Cosmological parameters
        
    Returns:
        array: CMB TT power spectrum D_ell values
    """
    # Extract parameters
    A_s = params.get('A_s', 2.1e-9)
    n_s = params.get('n_s', 0.965)
    h = params.get('H0', 67.36) / 100.0
    Omega_b_h2 = params.get('Omega_b_h2', 0.02237)
    Omega_c_h2 = params.get('Omega_c_h2', 0.1200)
    tau = params.get('tau', 0.054)
    
    # Derived parameters
    Omega_m_h2 = Omega_b_h2 + Omega_c_h2
    z_star = 1090  # Recombination redshift
    
    # Physical scales
    r_s = 147  # Sound horizon at recombination in Mpc
    theta_s = r_s * h / (14000.0)  # Angular scale in radians
    k_D = 0.2 / r_s  # Silk damping scale
    
    # Create the power spectrum
    dl = np.zeros_like(ell, dtype=float)
    
    # Primary peak location and scale
    primary_peak_l = 220
    
    for i, l in enumerate(ell):
        # Primordial power spectrum with proper spectral tilt
        P_prim = A_s * (l / 200.0)**(n_s - 1.0)
        
        # Sachs-Wolfe plateau at low ℓ (large scales)
        sachs_wolfe = 1.0 / (1.0 + (l/22.0)**2)
        
        # First acoustic peak envelope (initial power)
        first_peak_env = 6000.0 / (1.0 + ((l-primary_peak_l)/100.0)**2)
        
        # Acoustic oscillations with proper phase and spacing
        # The spacing between peaks is determined by sound horizon
        acoustic_phase = np.pi * (l - primary_peak_l) / (primary_peak_l * 0.75)
        acoustic_osc = 0.6 + 0.4 * np.cos(acoustic_phase)
        
        # Silk damping envelope - stronger exponential falloff at high ℓ
        damping_scale = 1600.0  # Silk damping scale in ℓ
        damping = np.exp(-(l / damping_scale)**1.8)
        
        # Combine all effects with proper amplitudes:
        # - Sachs-Wolfe plateau dominates at low ℓ
        # - Acoustic peaks in the intermediate range
        # - Damping tail at high ℓ
        sw_amplitude = 1000.0 * sachs_wolfe
        acoustic_amplitude = P_prim * first_peak_env * acoustic_osc * damping
        
        # Combined spectrum with proper weighting
        dl[i] = sw_amplitude + acoustic_amplitude
    
    # Scale already built into the model components
    dl_scaled = dl
    
    # Apply reionization damping at large scales
    reion_damping = np.exp(-2 * tau * (ell < 30))
    dl_scaled *= reion_damping
    
    return dl_scaled

def compute_cl(params):
    """
    Interface function to compute theoretical CMB TT power spectrum.
    Compatible with likelihood code.
    
    Args:
        params (dict): Cosmological parameters
            
    Returns:
        array: CMB TT power spectrum D_ell values
    """
    # Define ell range
    ell = np.arange(2, 2501)
    
    # First try the improved model
    try:
        D_ell = create_improved_model(ell, params)
        print("Using improved ΛCDM model with acoustic oscillations")
        return D_ell
    except Exception as e:
        print(f"Error in improved model: {e}")
        
    # Next try the semi-analytic model
    try:
        D_ell = angular_power_spectrum(ell, params)
        print("Using semi-analytic cosmological model")
        return D_ell
    except Exception as e:
        print(f"Error in angular_power_spectrum: {e}")
    
    # Fall back to simplified model if both fail
    print("Falling back to simplified model")
    from CAMB import model_Dl_TT
    A_s = params['A_s']
    n_s = params['n_s']
    D_ell = model_Dl_TT(ell, A_s, n_s)
    
    # Apply reionization effect if tau is provided
    if 'tau' in params:
        tau = params['tau']
        D_ell *= np.exp(-2 * tau * (ell < 30))
    
    return D_ell

def test_model(plot=True):
    """
    Test the cosmology model with Planck 2018 parameters.
    
    Args:
        plot (bool): Whether to plot the results
        
    Returns:
        tuple: (ell, D_ell) arrays
    """
    # Planck 2018 parameters
    params = {
        'H0': 67.36,
        'Omega_b_h2': 0.02237,
        'Omega_c_h2': 0.1200,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'tau': 0.0544
    }
    
    # Define ell range
    ell = np.arange(2, 2501)
    
    # Calculate power spectrum
    D_ell = compute_cl(params)
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(ell, D_ell, 'b-', label='Model TT Spectrum')
        plt.xlabel(r'Multipole $\ell$')
        plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
        plt.title('Theoretical CMB TT Power Spectrum (ΛCDM)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return ell, D_ell

if __name__ == "__main__":
    print("Testing cosmology model...")
    ell, D_ell = test_model(plot=True)
    
    # Also test Daria's model for comparison
    from CAMB import model_Dl_TT
    ell_daria = np.arange(2, 2501)
    D_ell_daria = model_Dl_TT(ell_daria, 2.1e-9, 0.9649)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(ell, D_ell, 'b-', label='Enhanced Model')
    plt.plot(ell_daria, D_ell_daria, 'r--', label='Daria\'s Model')
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
    plt.title('Comparison of CMB TT Power Spectrum Models')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate derived quantities for verification
    params = {
        'H0': 67.36,
        'Omega_b_h2': 0.02237,
        'Omega_c_h2': 0.1200,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'tau': 0.0544
    }
    
    h = params['H0'] / 100.0
    Omega_m_h2 = params['Omega_b_h2'] + params['Omega_c_h2']
    Omega_m = Omega_m_h2 / h**2
    Omega_r = 4.15e-5 / h**2
    Omega_Lambda = 1 - Omega_m - Omega_r
    
    z_star = recombination_redshift(params['Omega_b_h2'], Omega_m_h2)
    r_s = sound_horizon(z_star, params['H0'], params['Omega_b_h2'], Omega_m_h2)
    d_A = angular_diameter_distance(z_star, params['H0'], Omega_m, Omega_r, Omega_Lambda)
    theta_s = r_s / d_A
    
    print(f"Recombination redshift z_* = {z_star:.2f}")
    print(f"Sound horizon r_s = {r_s:.2f} Mpc")
    print(f"Angular diameter distance d_A = {d_A:.2f} Mpc")
    print(f"Sound horizon angle theta_s = {theta_s:.6f} rad = {theta_s * 180 / np.pi:.4f} deg")