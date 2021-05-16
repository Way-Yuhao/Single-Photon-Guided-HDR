from math import exp, log, sqrt
import numpy as np
import matplotlib.pyplot as plt


"""SPAD constants"""
# flux = 1e7  # true incident photon flux [arbitrary]
T = 0.001    # exposure time [arbitrary]
q = .4  # quantum efficiency [experimental, Sup 7]
tau = 149.7e-9  # dead time (s)
dark_count = 100  # dark count rate (photons/s) [experimental, Sup 7]
p_ap = .01  # after-pulsing probability [exp, Sup 7]

"CCD constants"
fwc = 33400  # full well capacity (electrons) [Sup 7]
q_cmos = .9  # quantum efficiency [Sup 7]
sigma_r = 5  # read noise power (electrons) [Sup 7]


def spad_snr_0(flux):
    a = ((dark_count / flux) + q * (1 + flux * tau) * p_ap * exp(-1 * q * flux * tau))**2
    b = (1 + q * flux * tau) / (q * flux * T)
    c = ((1 + q * flux * tau)**4) / (12 * q**2 * flux**2 * T**2)
    snr = -10 * log(10, (a + b + c))
    return snr


def spad_snr3(flux):
    B_ap = p_ap * q * flux * (1 + flux * tau) * np.exp(-1 * q * flux * tau)
    V_shot = (flux * (1 + q * flux * tau)) / (q * T)
    V_quanti = ((1 + q * flux * tau)**4) / (12 * q**2 * T**2)
    RMSE = np.sqrt((dark_count + B_ap)**2 + V_shot + V_quanti)
    SNR = 20 * np.log(10, (flux / RMSE))
    return SNR


def spad_snr(flux):
    B_ap = p_ap * q * flux * (1 + flux * tau) * exp(-1 * q * flux * tau)
    V_shot = (flux * (1 + q * flux * tau)) / (q * T)
    V_quanti = ((1 + q * flux * tau)**4) / (12 * q**2 * T**2)
    RMSE = sqrt((dark_count + B_ap)**2 + V_shot + V_quanti)
    SNR = 20 * log((flux / RMSE), 10)
    return SNR


def ccd_snr(flux):
    if flux < (fwc / (q_cmos * T)):
        SNR = 10 * log((q_cmos**2 * flux**2 * T**2)/(q_cmos * flux * T + sigma_r**2), 10)
    else:
        SNR = float('-10000')
    return SNR


def disp_spad():
    x = np.linspace(1e2, 1e11, num=5000)
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i] = spad_snr(x[i])
    plt.plot(x, y, label="SPAD, T={} s".format(T))


def disp_ccd():
    x = np.linspace(1e2, 1e11, num=5000)
    y2 = np.zeros(x.shape)
    for i in range(len(y2)):
        y2[i] = ccd_snr(x[i])
    plt.plot(x, y2, label="CCD, T={} s".format(T))


def main():
    disp_spad()
    disp_ccd()
    plt.xscale("log")
    plt.ylim(top=50, bottom=-40)
    plt.title("Theoretical SNR vs Incident Flux")
    plt.ylabel("theoretical SNR (dB)")
    plt.xlabel("incident photon flux")
    plt.legend(loc="lower right")
    plt.show()



if __name__ == "__main__":
    main()
