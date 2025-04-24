# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 15:20:45 2025

@authors: Daria Teodora Harabor, David Turturean
"""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Constants
c = 299792.458  # Light speed (km/s)
G = 6.67430e-11  # Newton's G (m^3/kg/s^2)
h_planck = 6.62607015e-34  # Planck const (J.s)
k_B = 1.380649e-23  # Boltzmann (J/K)
sigma_T = 6.6524587321e-29  # Thomson x-section (m^2)
Mpc_to_m = 3.08567758e22  # Mpc to m conversion

# CMB stuff
T_cmb = 2.7255  # CMB temp (K)
rho_gamma = (np.pi**2 / 15) * (k_B * T_cmb)**4 / (c**3 * h_planck**3)  # Radiation density

# Pivot scale 
k_pivot = 0.05  # Mpc^-1

def hubble_parameter(z, H0, Omega_m, Omega_r, Omega_Lambda):
    """
    Get H(z) at redshift z.
    
    Args:
        z: redshift
        H0: Hubble constant (km/s/Mpc)
        Omega_m: matter density
        Omega_r: radiation density
        Omega_Lambda: dark energy density
        
    Returns:
        H(z) in km/s/Mpc
    """
    return H0 * np.sqrt(Omega_r * (1 + z)**4 + Omega_m * (1 + z)**3 + Omega_Lambda)

def comoving_distance(z, H0, Omega_m, Omega_r, Omega_Lambda):
    """
    Get comoving distance to z.
    
    Args:
        z: redshift
        H0: Hubble constant
        Omega_m: matter density
        Omega_r: radiation density
        Omega_Lambda: dark energy density
        
    Returns:
        comoving distance (Mpc)
    """
    def integrand(z_prime):
        return c / hubble_parameter(z_prime, H0, Omega_m, Omega_r, Omega_Lambda)
    
    result, _ = quad(integrand, 0, z)
    return result

def angular_diameter_distance(z, H0, Omega_m, Omega_r, Omega_Lambda):
    """
    Get angular diameter distance to z.
    
    Args:
        z: redshift
        H0: Hubble constant
        Omega_m: matter density
        Omega_r: radiation density
        Omega_Lambda: dark energy density
        
    Returns:
        angular diameter distance (Mpc)
    """
    d_C = comoving_distance(z, H0, Omega_m, Omega_r, Omega_Lambda)
    return d_C / (1 + z)

def sound_horizon(z_star, H0, Omega_b_h2, Omega_m_h2):
    """
    Get sound horizon at recombination.
    
    Args:
        z_star: recombination redshift
        H0: Hubble constant
        Omega_b_h2: physical baryon density
        Omega_m_h2: physical matter density
        
    Returns:
        sound horizon (Mpc)
    """
    # Convert params
    h = H0 / 100.0
    Omega_b = Omega_b_h2 / h**2
    Omega_m = Omega_m_h2 / h**2
    Omega_r = 4.15e-5 / h**2  # Radiation (γ+ν)
    Omega_Lambda = 1 - Omega_m - Omega_r  # Flat universe
    
    def integrand(z_prime):
        # Sound speed
        R = 3.0 * Omega_b / (4.0 * Omega_r) * (1 + z_prime)**(-1)
        cs = c / np.sqrt(3 * (1 + R))
        
        return cs / hubble_parameter(z_prime, H0, Omega_m, Omega_r, Omega_Lambda)
    
    result, _ = quad(integrand, z_star, np.inf, limit=1000)
    return result

def recombination_redshift(Omega_b_h2, Omega_m_h2):
    """
    Get z_* using Hu & Sugiyama formula.
    
    Args:
        Omega_b_h2: physical baryon density
        Omega_m_h2: physical matter density
        
    Returns:
        z_* (recombination redshift)
    """
    # Hu & Sugiyama (1996) formula
    g1 = 0.0783 * Omega_b_h2**(-0.238) / (1 + 39.5 * Omega_b_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Omega_b_h2**1.81)
    
    return 1048 * (1 + 0.00124 * Omega_b_h2**(-0.738)) * (1 + g1 * Omega_m_h2**g2)

def primordial_power_spectrum(k, A_s, n_s):
    """
    Get primordial power spectrum.
    
    Args:
        k: wavenumber (Mpc^-1)
        A_s: amplitude at pivot scale
        n_s: spectral index
        
    Returns:
        P(k) 
    """
    return A_s * (k / k_pivot)**(n_s - 1)

def transfer_function_bbks(k, Omega_m_h2, Omega_b_h2, h):
    """
    BBKS transfer function with baryon correction.
    
    Args:
        k: wavenumber (h/Mpc)
        Omega_m_h2: matter density
        Omega_b_h2: baryon density
        h: H0/100
        
    Returns:
        T(k)
    """
    # Convert units
    k_phys = k * h  # To Mpc^-1
    
    # Equality scale
    Omega_m = Omega_m_h2 / h**2
    z_eq = 25000 * Omega_m * h**2 / T_cmb**4
    k_eq = 0.073 * Omega_m * h**2 * T_cmb**(-2)  # Mpc^-1
    
    # BBKS formula
    q = k_phys / (k_eq * h)
    
    # Baryon stuff
    alpha_b = 2.07 * k_eq * h * (1 + R_drag(Omega_b_h2, Omega_m_h2))**(-3/4)
    beta_b = 0.5 + (Omega_b_h2/Omega_m_h2) / 3
    
    # BBKS function
    x = q * np.sqrt(alpha_b)
    T_bbks = np.log(1 + 2.34 * x) / (2.34 * x) * \
             (1 + 3.89 * x + (16.1 * x)**2 + (5.46 * x)**3 + (6.71 * x)**4)**(-0.25)
    
    # Add baryon effects
    T_baryon = T_bbks * (1 + (beta_b - 1) * (1 + (k_phys/5)**2)**(-1))
    
    return T_baryon

def R_drag(Omega_b_h2, Omega_m_h2):
    """
    Get R_drag for baryon effects.
    
    Args:
        Omega_b_h2: baryon density
        Omega_m_h2: matter density
        
    Returns:
        R_drag value
    """
    # Quick approximation
    return 31.5 * Omega_b_h2 / T_cmb**4 * (1000 / recombination_redshift(Omega_b_h2, Omega_m_h2))

def radiation_transfer_function(k, ell, tau, theta_s):
    """
    Basic CMB radiation transfer function.
    Includes:
    - Acoustic peaks
    - Silk damping
    - Reionization effects
    
    Args:
        k: wavenumber (Mpc^-1)
        ell: multipole
        tau: optical depth
        theta_s: sound horizon angle (rad)
        
    Returns:
        Transfer function
    """
    # ell ~ k*r (r = angular diameter distance)
    ell_k = ell
    
    # Acoustic part
    acoustic = np.sin(np.pi * ell / (theta_s * 180 / np.pi))**2
    
    # Silk damping
    k_silk = 0.2 * Mpc_to_m
    ell_silk = 1800
    envelope = np.exp(-(ell / ell_silk)**2)
    
    # Reionization
    reion_damping = np.exp(-2 * tau * (ell < 30))
    
    # All together
    T_rad = acoustic * envelope * reion_damping
    
    return T_rad

def angular_power_spectrum(ell, params):
    """
    Get CMB TT power spectrum.
    
    Args:
        ell: multipoles
        params: cosmo params dict with:
            - H0: Hubble constant
            - Omega_b_h2: baryon density
            - Omega_c_h2: CDM density
            - n_s: spectral index
            - A_s: amplitude
            - tau: optical depth
            
    Returns:
        D_ell values (μK^2)
    """
    # Get params
    H0 = params['H0']
    h = H0 / 100.0
    Omega_b_h2 = params['Omega_b_h2']
    Omega_c_h2 = params['Omega_c_h2']
    Omega_m_h2 = Omega_b_h2 + Omega_c_h2
    n_s = params['n_s']
    A_s = params['A_s']
    tau = params['tau']
    
    # Get z_star
    z_star = recombination_redshift(Omega_b_h2, Omega_m_h2)
    
    # Get sound horizon
    r_s = sound_horizon(z_star, H0, Omega_b_h2, Omega_m_h2)
    
    # Get angular diameter distance
    Omega_m = Omega_m_h2 / h**2
    Omega_r = 4.15e-5 / h**2
    Omega_Lambda = 1 - Omega_m - Omega_r
    d_A = angular_diameter_distance(z_star, H0, Omega_m, Omega_r, Omega_Lambda)
    
    # Sound horizon angle
    theta_s = r_s / d_A * 180 / np.pi  # degrees
    
    # Approx ℓ to k conversion
    k_array = ell / d_A
    
    # Get primordial power
    P_prim = primordial_power_spectrum(k_array, A_s, n_s)
    
    # Apply transfer function
    T_rad = radiation_transfer_function(k_array, ell, tau, theta_s)
    
    # Get C_ell
    C_ell = P_prim * T_rad
    
    # Convert to D_ell = ℓ(ℓ+1)C_ℓ/2π
    D_ell = ell * (ell + 1) * C_ell / (2 * np.pi)
    
    # To μK^2
    D_ell *= 1e9
    
    return D_ell

def create_improved_model(ell, params):
    """
    Better ΛCDM model for CMB TT spectrum.
    
    Not as good as full Boltzmann code but has all the main features.
    
    Args:
        ell: multipoles
        params: cosmo params
        
    Returns:
        D_ell values
    """
    # Get params
    A_s = params.get('A_s', 2.1e-9)
    n_s = params.get('n_s', 0.965)
    h = params.get('H0', 67.36) / 100.0
    Omega_b_h2 = params.get('Omega_b_h2', 0.02237)
    Omega_c_h2 = params.get('Omega_c_h2', 0.1200)
    tau = params.get('tau', 0.054)
    
    # Derived stuff
    Omega_m_h2 = Omega_b_h2 + Omega_c_h2
    
    # Get z_star - Hu & Sugiyama formula
    g1 = 0.0783 * Omega_b_h2**(-0.238) / (1 + 39.5 * Omega_b_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Omega_b_h2**1.81)
    z_star = 1048 * (1 + 0.00124 * Omega_b_h2**(-0.738)) * (1 + g1 * Omega_m_h2**g2)
    
    # Get sound horizon
    r_s = sound_horizon(z_star, params['H0'], Omega_b_h2, Omega_m_h2)
    
    # Get d_A to last scattering
    Omega_m = Omega_m_h2 / h**2
    Omega_r = 4.15e-5 / h**2
    Omega_Lambda = 1 - Omega_m - Omega_r
    d_A = angular_diameter_distance(z_star, params['H0'], Omega_m, Omega_r, Omega_Lambda)
    
    # Acoustic scale ℓ_A = π·d_A/r_s
    ell_A = np.pi * d_A / r_s
    
    # Sound horizon angle
    theta_s = r_s / d_A  # radians
    
    # Silk damping scale
    ell_D = 1600.0 * (Omega_b_h2/0.02237)**(-0.25) * (Omega_m_h2/0.1424)**(-0.125)
    
    # Power spectrum array
    dl = np.zeros_like(ell, dtype=float)
    
    # First peak position - should be around ell_A
    primary_peak_l = ell_A
    
    for i, l in enumerate(ell):
        # Primordial power with tilt
        P_prim = A_s * (l / 200.0)**(n_s - 1.0)
        
        # SW plateau (large scales)
        sachs_wolfe = 1.0 / (1.0 + (l/22.0)**2)
        
        # First peak shape
        first_peak_env = 6000.0 / (1.0 + ((l-primary_peak_l)/100.0)**2)
        
        # Acoustic oscillations
        acoustic_phase = np.pi * (l - primary_peak_l) / (primary_peak_l * 0.75)
        acoustic_osc = 0.6 + 0.4 * np.cos(acoustic_phase)
        
        # Silk damping - exp(-ℓ²/ℓ_D²) as in main.tex
        damping = np.exp(-(l / ell_D)**2)
        
        # Put it all together:
        # SW at low ℓ, acoustic peaks in middle, damping at high ℓ
        sw_amplitude = 1000.0 * sachs_wolfe
        acoustic_amplitude = P_prim * first_peak_env * acoustic_osc * damping
        
        # Final spectrum
        dl[i] = sw_amplitude + acoustic_amplitude
    
    # Already scaled
    dl_scaled = dl
    
    # Reionization effects at large scales
    reion_damping = np.exp(-2 * tau * (ell < 30))
    dl_scaled *= reion_damping
    
    return dl_scaled

def compute_cl(params):
    """
    Get CMB TT power spectrum - main interface function.
    
    Args:
        params: cosmo parameters
            
    Returns:
        D_ell values
    """
    # Set ell range
    ell = np.arange(2, 2501)
    
    # Try best model first
    try:
        D_ell = create_improved_model(ell, params)
        print("Using better ΛCDM model")
        return D_ell
    except Exception as e:
        print(f"Error: {e}")
        
    # Try simpler model next
    try:
        D_ell = angular_power_spectrum(ell, params)
        print("Using semi-analytic model")
        return D_ell
    except Exception as e:
        print(f"Error: {e}")
    
    # Last resort
    print("Using basic model")
    from CAMB import model_Dl_TT
    A_s = params['A_s']
    n_s = params['n_s']
    D_ell = model_Dl_TT(ell, A_s, n_s)
    
    # Add reionization
    if 'tau' in params:
        tau = params['tau']
        D_ell *= np.exp(-2 * tau * (ell < 30))
    
    return D_ell

def test_model(plot=True):
    """
    Test model with Planck 2018 params.
    
    Args:
        plot: show plot or not
        
    Returns:
        ell and D_ell values
    """
    # Planck 2018 best-fit
    params = {
        'H0': 67.36,
        'Omega_b_h2': 0.02237,
        'Omega_c_h2': 0.1200,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'tau': 0.0544
    }
    
    # ell values
    ell = np.arange(2, 2501)
    
    # Get spectrum
    D_ell = compute_cl(params)
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(ell, D_ell, 'b-', label='TT Spectrum')
        plt.xlabel(r'$\ell$')$')
        plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
        plt.title('CMB TT Power Spectrum (ΛCDM)')
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