from math import exp, log, sqrt
import numpy as np
import matplotlib.pyplot as plt


# """SPAD constants"""
# T = .001    # exposure time (s) [arbitrary]
# q = .4  # quantum efficiency [experimental, Sup 7]
# tau = 149.7e-9  # dead time (s)
# dark_count = 100  # dark count rate (photons/s) [experimental, Sup 7]
# p_ap = .01  # after-pulsing probability [exp, Sup 7]
#
# "CCD constants"
# T_short = 0.0001
# T_long = 0.001
# fwc = 33400  # full well capacity (electrons) [Sup 7]
# q_cmos = .9  # quantum efficiency [Sup 7]
# sigma_r = 5  # read noise power (electrons) [Sup 7]


"""SPAD constants"""
T = .01    # exposure time (s) [arbitrary]
q = .4  # quantum efficiency [experimental, Sup 7]
tau = 149.7e-9  # dead time (s)
dark_count = 0  # dark count rate (photons/s) [experimental, Sup 7]
p_ap = 0  # after-pulsing probability [exp, Sup 7]

"CCD constants"
T_short = 0.001
T_long = 0.01
fwc = 2**12  # full well capacity (electrons) [Sup 7]
q_cmos = .7  # quantum efficiency [Sup 7]
sigma_r = 0  # read noise power (electrons) [Sup 7]



def spad_snr(flux):
    B_ap = p_ap * q * flux * (1 + flux * tau) * exp(-1 * q * flux * tau)
    V_shot = (flux * (1 + q * flux * tau)) / (q * T)
    V_quanti = ((1 + q * flux * tau)**4) / (12 * q**2 * T**2)
    RMSE = sqrt((dark_count + B_ap)**2 + V_shot + V_quanti)
    SNR = 20 * log((flux / RMSE), 10)
    return SNR


def ccd_snr(flux, T_ccd):
    if flux < (fwc / (q_cmos * T_ccd)):
        SNR = 10 * log((q_cmos ** 2 * flux ** 2 * T_ccd ** 2) / (q_cmos * flux * T_ccd + sigma_r ** 2), 10)
    else:
        SNR = -1000000
    return SNR


def disp_spad():
    x = np.logspace(2, 11, 100)
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i] = spad_snr(x[i])
    plt.plot(x, y, label="SPAD, T={} s".format(T))


def disp_ccd():
    x = np.logspace(2, 11, 100)
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i] = ccd_snr(x[i], T)
    plt.plot(x, y, "orange", label="CCD, T={} s".format(T))


def dual_ccd():
    x = np.logspace(2, 11, 100)
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y1 = ccd_snr(x[i], T_long)
        y2 = ccd_snr(x[i], T_short)
        y[i] = max(y1, y2)
    plt.plot(x, y, "orange", label="Dual CCDs, T = {}s & {} s".format(T_long, T_short))


def ccd_n_spad():
    x = np.logspace(2, 11, 100)
    y= np.zeros(x.shape)
    for i in range(len(x)):
        y1 = ccd_snr(x[i], T_long)
        y2 = spad_snr(x[i])
        y[i] = max(y1, y2)
    plt.plot(x, y, "green", label="CCD + SPAD Hybrid, both T={} s".format(T_long))


def main():
    # disp_spad()
    # disp_ccd()
    dual_ccd()
    ccd_n_spad()
    plt.xscale("log")
    plt.ylim(top=70, bottom=-20)
    plt.title("Theoretical SNR vs Incident Flux")
    plt.ylabel("theoretical SNR (dB)")
    plt.xlabel("incident photon flux")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()




# def spad_snr_0(flux):
#     a = ((dark_count / flux) + q * (1 + flux * tau) * p_ap * exp(-q * flux * tau))**2
#     b = (1 + q * flux * tau) / (q * flux * T)
#     c = ((1 + q * flux * tau)**4) / (12 * q**2 * flux**2 * T**2)
#     snr = -10 * log((a + b + c), 10)
#     return snr