from math import exp, log
import numpy as np
import matplotlib.pyplot as plt

# define constants

# flux = 1e7  # true incident photon flux [arbitrary]
T = .01    # exposure time [arbitrary]
q = .4  # quantum efficiency [experimental, Sup 7]
tau = 149.7e-9  # dead time (s)
dark_count = 100  # dark count rate (photons/s) [experimental, Sup 7]
p_ap = .01  # after-pulsing probability [exp, Sup 7]


def spad_snr(flux):
    a = (dark_count / flux + q * (1 + flux * tau) * p_ap * np.exp(-q * flux * tau))**2
    b = (1 + q * flux * tau) / (q * flux * T)
    c = ((1 + q * flux * tau)**4) / (12 * q**2 * flux**2 * T**2)
    snr = -10 * np.log(10, (a + b + c))
    return snr


def main():
    x = np.linspace(1e2, 1e11, num=50)
    plt.plot(x, spad_snr(x))
    plt.show()


if __name__ == "__main__":
    main()
