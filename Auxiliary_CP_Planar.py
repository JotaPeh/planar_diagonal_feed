from scipy import special as sp
from scipy.optimize import root_scalar
from scipy import integrate as spi
from scipy.interpolate import interp1d
import numpy as np
from functools import partial
import warnings
import time

# Planar (rectangular, near-square) counterpart of the spherical Auxiliary_CP.py.
# Mapping used (theta -> x, phi -> y):
#   Dthetaa, Dphia  -> Dxa, Dya       (physical patch size along x and y)
#   thetap, phip    -> xp, yp         (feed position relative to the patch center)
#   deltatheta/Phi  -> deltax, deltay (fringing extension of the cavity)
#   Dphip           -> Dxp            (width of the current ribbon that models the probe)
#   phip_add        -> removed
# The Legendre eigenfunctions become cosines, so the eigenvalue root searches
# (Theta_find / Phi_find) become closed form: k10 = pi/Dx and k01 = pi/Dy.
# Higher-order modes remain lumped into the probe reactance Xp (times Yp).

def Z_par_g(Dxp, Yp, tg01, tg10, deltaxc, deltayc, Dxa, Dya, xp, yp, sweep, geom = [2.55, 1.524e-3]):
    # General constants
    e0 = 8.854e-12          # (F/m)
    u0 = np.pi*4e-7         # (H/m)
    gamma = 0.577216        # Euler-Mascheroni constant
    eps = 1e-5              # Numerical error threshold

    # Cavity geometry
    er, h = geom
    #er = 2.55               # Substrate relative permittivity
    #h = 1.524e-3            # Substrate thickness (m)

    x1, x2 = -Dxa/2, Dxa/2          # Physical patch edges along x (m)
    y1, y2 = -Dya/2, Dya/2          # Physical patch edges along y (m)

    x1c, x2c = x1 - deltaxc, x2 + deltaxc      # Cavity walls along x (m)
    y1c, y2c = y1 - deltayc, y2 + deltayc      # Cavity walls along y (m)

    # Coaxial feed coordinates (m)
    xp = [xp]
    yp = [yp]
    df = 1.3e-3             # Coaxial center pin diameter (m)
    es = e0 * er            # Substrate permittivity (F/m)

    # Cavity mode function along x (magnetic walls at x1c and x2c)
    def R(x, Ux):
        return np.cos(Ux * (x - x1c))

    def R2(x, Ux):
        return R(x, Ux)**2

    # Resonant frequencies (Hz)
    f01 = 1 / (2 * (y2c - y1c) * np.sqrt(es * u0))
    f10 = 1 / (2 * (x2c - x1c) * np.sqrt(es * u0))

    k01 = 2*np.pi*f01*np.sqrt(u0*es)
    k10 = 2*np.pi*f10*np.sqrt(u0*es)

    # Input impedance - RLC circuit:
    def tgefTot(Ux, Uy):
        if Uy == 0:
            return tg10
        else:
            return tg01

    def RLC(f, klm, Ux, Uy, p1, p2):
        Ilm = spi.quad(partial(R2, Ux = Ux), x1c, x2c)[0]

        # Auxiliar alpha
        alpha = h * (R(xp[p1-1], Ux) * np.cos(Uy * (yp[p1-1]-y1c)) * np.sinc(Ux * Dxp / (2*np.pi)) * np.sinc(Uy * Dxp / (2*np.pi))) * \
            (R(xp[p2-1], Ux) * np.cos(Uy * (yp[p2-1]-y1c)) * np.sinc(Ux * Dxp / (2*np.pi)) * np.sinc(Uy * Dxp / (2*np.pi))) / ((y2c - y1c) * es * Ilm)
        if Uy != 0:
            alpha = 2 * alpha

        # RLC components
        RLM = alpha/(2 * np.pi * f * tgefTot(Ux, Uy))
        CLM = 1/alpha
        LLM = alpha/(2 * np.pi * klm / (2*np.pi*np.sqrt(u0*es)))**2
        return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM)))

    def ZCP(f, p1, p2):
        k = (2 * np.pi * f) * np.sqrt(es * u0)
        eta = np.sqrt(u0 / es)
        Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
        return RLC(f, k01, 0, np.pi/(y2c-y1c), p1, p2) + RLC(f, k10, np.pi/(x2c-x1c), 0, p1, p2) + 1j*Xp*Yp

    return ZCP(np.linspace(sweep[0], sweep[1], sweep[2]), 1, 1)

def synth_ant(flm_des, geom = [2.55, 1.524e-3]):
    # General constants
    dtr = np.pi/180         # Degrees to radians (rad/deg)
    e0 = 8.854e-12          # (F/m)
    u0 = np.pi*4e-7         # (H/m)
    c = 1/np.sqrt(e0*u0)    # Speed of light in vacuum (m/s)
    gamma = 0.577216        # Euler-Mascheroni constant
    eps = 1e-5              # Numerical error threshold

    # Cavity geometry
    er, h = geom
    #er = 2.55               # Substrate relative permittivity
    #h = 1.524e-3            # Substrate thickness (m)

    # Coaxial feed
    df = 1.3e-3             # Coaxial center pin diameter (m)
    es = e0 * er            # Substrate permittivity (F/m)

    # Other variables
    tgdel = 0.0022          # Dielectric loss tangent
    sigma = 5.8e7          # Conductor electric conductivity (S/m)

    # Initial estimate of the fringing extension (Hammerstad). The cavity is filled
    # with er (as in the spherical model), so the extension is the one that makes
    # the extended cavity resonate where the Hammerstad patch does. It is only a
    # first guess: deltax and deltay are calibrated afterwards against the HFSS.
    Lc0 = np.pi / (2*np.pi*flm_des*np.sqrt(u0*es))
    L0 = Lc0
    for _ in range(30):
        eeff = (er+1)/2 + (er-1)/2 / np.sqrt(1 + 12*h/L0)
        dL = 0.412*h*(eeff+0.3)*(L0/h+0.264) / ((eeff-0.258)*(L0/h+0.8))
        L0 = c/(2*flm_des*np.sqrt(eeff)) - 2*dL
    deltax = deltay = np.max([(Lc0 - L0)/2, 1e-4])      # Fringing extension (m)

    # Cavity mode function along x
    def R(x, Ux, x1c):
        return np.cos(Ux * (x - x1c))

    def R2(x, Ux, x1c):
        return R(x, Ux, x1c)**2

    #########################################################################################################################################################
    # Closed-form replacements of the root searches (separable cavity modes)
    def Dx_find(K):
        return np.pi/K       # TM_10 mode: K = pi/Dx

    def Dy_find(K):
        return np.pi/K       # TM_01 mode: K = pi/Dy

    #########################################################################################################################################################
    # Iterative design
    def RLC(f, Kex, Ux, Uy, xp, yp, tgef, Dx, Dy):
        Dxp = np.exp(1.5)*df/2
        x1c, x2c = -Dx/2, Dx/2
        y1c, y2c = -Dy/2, Dy/2
        Ilm = spi.quad(partial(R2, Ux = Ux, x1c = x1c), x1c, x2c)[0]

        alpha = h * (R(xp, Ux, x1c) * np.cos(Uy * (yp-y1c)) * np.sinc(Ux * Dxp / (2*np.pi)) * np.sinc(Uy * Dxp / (2*np.pi))) * \
            (R(xp, Ux, x1c) * np.cos(Uy * (yp-y1c)) * np.sinc(Ux * Dxp / (2*np.pi)) * np.sinc(Uy * Dxp / (2*np.pi))) / ((y2c - y1c) * es * Ilm)
        if Uy != 0:
            alpha = 2 * alpha

        RLM = alpha/(2 * np.pi * f * tgef)
        CLM = 1/alpha
        LLM = alpha/(2 * np.pi * Kex / (2*np.pi*np.sqrt(u0*es)))**2
        return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM)))

    def Z(f, k01, k10, xp, yp, tgef01, tgef10, Dx, Dy):
        Ux01, Uy01 = 0, np.pi/Dy
        Ux10, Uy10 = np.pi/Dx, 0
        k = (2 * np.pi * f) * np.sqrt(es * u0)
        eta = np.sqrt(u0 / es)
        Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
        return RLC(f, k01, Ux01, Uy01, xp, yp, tgef01, Dx, Dy) + RLC(f, k10, Ux10, Uy10, xp, yp, tgef10, Dx, Dy) + 1j*Xp

    def Grad(k0, Ls, d):
        # Radiation conductance (S) of two in-phase slots of length Ls separated by d
        # (slot voltage = E*h), including the mutual term
        a = k0*Ls/2
        I1 = spi.quad(lambda th: (a*np.sinc(a*np.cos(th)/np.pi))**2 * np.sin(th)**3, 0, np.pi)[0]
        I12 = spi.quad(lambda th: (a*np.sinc(a*np.cos(th)/np.pi))**2 * sp.j0(k0*d*np.sin(th)) * np.sin(th)**3, 0, np.pi)[0]
        return 2*(I1 + I12)/(120*np.pi**2)

    def tgefTot(klm, mode, Dx, Dy):
        Rs = np.sqrt(klm * np.sqrt(u0/es) / (2 * sigma))
        Qc = klm * np.sqrt(u0/es) * h / (2 * Rs)

        Dxa, Dya = Dx - 2*deltax, Dy - 2*deltay
        w = klm / np.sqrt(u0*es)         # Resonant angular frequency (rad/s)
        K0 = klm / np.sqrt(er)

        if mode == '10':
            # TM_10: radiating slots at the x edges, length Dya, separation Dx
            Q10 = w * es * Dx * Dy / (2 * Grad(K0, Dya, Dx) * h)
            return tgdel + 1/Qc + 1/Q10
        elif mode == '01':
            # TM_01: radiating slots at the y edges, length Dxa, separation Dy
            Q01 = w * es * Dx * Dy / (2 * Grad(K0, Dxa, Dy) * h)
            return tgdel + 1/Qc + 1/Q01

    def S(Kdes, Dx, Dy):
        # Ratio between the broadside fields radiated by TM_10 (x polarized, slots at
        # the x edges, length Dya) and TM_01 (y polarized, slots at the y edges,
        # length Dxa), per unit mode amplitude at the cavity walls. At broadside
        # there is no phase term (flat aperture), so S is real.
        Dxa, Dya = Dx - 2*deltax, Dy - 2*deltay
        return (2 * h * Dya) / (2 * h * Dxa) + 0j

    def Probe(k01, k10, Kef01, Kef10, Dx, Dy):
        U10, U01 = np.pi/Dx, np.pi/Dy
        x1c, x2c = -Dx/2, Dx/2
        y1c = -Dy/2
        x1, x2 = x1c + deltax, x2c - deltax

        I01 = spi.quad(partial(R2, Ux = 0, x1c = x1c), x1c, x2c)[0]
        I10 = spi.quad(partial(R2, Ux = U10, x1c = x1c), x1c, x2c)[0]

        xp = np.linspace(x1, x2, 101)

        V = (I01 * R(x1c, U10, x1c))/(2 * I10 * R((x1c+x2c)/2, 0, x1c)) * (Kef01**2 - k01**2)/(Kef10**2 - k10**2) * S(np.real(Kef01), Dx, Dy)
        yp = y1c + np.arccos(R(xp, U10, x1c) / R(xp, 0, x1c) * np.abs(V)) / U01

        mask = ~np.isnan(yp)
        yp = yp[mask]
        xp = xp[mask]

        return yp, xp

    # Design
    def imZin(p, Analysis = 0):
        k01 = k10 = 2*np.pi*flm_des*np.sqrt(u0*es)
        Dx = Dx_find(k10)
        Dy = Dy_find(k01)

        tgef01 = tgefTot(k01, '01', Dx, Dy)
        tgef10 = tgefTot(k10, '10', Dx, Dy)

        epsilon = 1
        tol = 1e-4
        steps = 0
        while epsilon > tol:
            steps += 1
            kl = k10 + p*(k01 - k10)
            k01ll = kl*tgef01/2
            k10ll = kl*tgef10/2
            phaseK = np.pi/2 - np.angle(S(kl, Dx, Dy))

            k01prev = k01
            k10prev = k10

            # Supose k01 > k10
            k10 = kl - ((k10ll+p*(k01ll-k10ll)) - np.sqrt((k10ll-p*(k01ll+k10ll))**2 - 4*k01ll*k10ll*(p-1)*p/np.cos(phaseK)**2) ) / (2*np.tan(phaseK)*(p-1))
            k01 = kl - ((k10ll+p*(k01ll-k10ll)) - np.sqrt((k10ll-p*(k01ll+k10ll))**2 - 4*k01ll*k10ll*(p-1)*p/np.cos(phaseK)**2) ) / (2*np.tan(phaseK)*p)
            if k10 > k01:
                k10 = kl - ((k10ll+p*(k01ll-k10ll)) + np.sqrt((k10ll-p*(k01ll+k10ll))**2 - 4*k01ll*k10ll*(p-1)*p/np.cos(phaseK)**2) ) / (2*np.tan(phaseK)*(p-1))
                k01 = kl - ((k10ll+p*(k01ll-k10ll)) + np.sqrt((k10ll-p*(k01ll+k10ll))**2 - 4*k01ll*k10ll*(p-1)*p/np.cos(phaseK)**2) ) / (2*np.tan(phaseK)*p)

            Dx = Dx_find(k10)
            Dy = Dy_find(k01)

            tgef01 = tgefTot(k01, '01', Dx, Dy)
            tgef10 = tgefTot(k10, '10', Dx, Dy)

            epsilon = np.max([np.abs(k01prev - k01), np.abs(k10prev - k10)])

            if steps > 50 or k10 is np.nan or k01 is np.nan:
                    return np.nan

        Py, Px = Probe(k01, k10, kl-1j*k01ll, kl-1j*k10ll, Dx, Dy)

        Pyx = interp1d(Px, Py)

        def Z50(x):
            return np.real(Z(flm_des, k01, k10, x, Pyx(x), tgef01, tgef10, Dx, Dy))-50
        # The bracket starts slightly off the center: both modes vanish at x = y = 0,
        # which makes alpha = 0 and RLC a 0/0 there (in the sphere R10(pi/2) != 0)
        root = root_scalar(Z50, bracket=[Px[-1]*1e-3, Px[-1]], method='bisect')

        xp = np.array(root.root)

        Zin = Z(flm_des, k01, k10, xp, Pyx(xp), tgef01, tgef10, Dx, Dy)

        if  Analysis:
            Dya = Dy - 2*deltay
            Dxa = Dx - 2*deltax
            return Dxa, Dya, xp, Pyx(xp), tgefTot(k01, '01', Dx, Dy), tgefTot(k10, '10', Dx, Dy)
        return np.imag(Zin)

    x = np.round(np.linspace(0.05, 0.95, 10), 2)
    y_list = []
    for xi in x:
        warnings.simplefilter("ignore")
        try:
            y_list.append(imZin(xi, Analysis=0))
        except:
            y_list.append(np.nan)
        # print('xi = ', xi, ' --> Zin imag = ', y_list[-1])
    y = np.array(y_list, dtype=float)
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError("Not enough points to find a root.")
    if np.all(y>0) or np.all(y<0):
        raise ValueError("No interval found where the function changes sign.")

    idx = 0
    for i in range(len(y)-1):
        if y[i] * y[i+1] <= 0:
            idx = i
            if x[i+1]-x[i]<0.11:
                break
            else:
                raise ValueError("Too large an interval between points of sign change.")

    Wrapper = partial(imZin, Analysis = 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = root_scalar(Wrapper, bracket=[x[idx], x[idx+1]], method='bisect', xtol = 1e-4)
        Dxa, Dya, xp, yp, tg01, tg10 = imZin(np.array(root.root), Analysis = 1)
    #print('p = ', np.array(root.root))

    parametros = np.array([Dxa, Dya, xp, yp])
    estimativas = np.array([np.exp(1.5)*df/2, 1.0, tg01, tg10, deltax, deltay])
    return parametros, estimativas

#########################################################################################################################################################
# Sythesis of the antenna after the parameters update
def synth_ant_pos(flm_des, estim, geom = [2.55, 1.524e-3]):
    Dxp, Yp, tg01, tg10, deltax, deltay = estim
    # General constants
    dtr = np.pi/180         # Degrees to radians (rad/deg)
    e0 = 8.854e-12          # (F/m)
    u0 = np.pi*4e-7         # (H/m)
    gamma = 0.577216        # Euler-Mascheroni constant
    eps = 1e-5              # Numerical error threshold

    # Cavity geometry
    er, h = geom
    #er = 2.55               # Substrate relative permittivity
    #h = 1.524e-3            # Substrate thickness (m)

    # Coaxial feed
    df = 1.3e-3             # Coaxial center pin diameter (m)
    es = e0 * er            # Substrate permittivity (F/m)

    # Cavity mode function along x
    def R(x, Ux, x1c):
        return np.cos(Ux * (x - x1c))

    def R2(x, Ux, x1c):
        return R(x, Ux, x1c)**2

    #########################################################################################################################################################
    # Closed-form replacements of the root searches (separable cavity modes)
    def Dx_find(K):
        return np.pi/K

    def Dy_find(K):
        return np.pi/K

    #########################################################################################################################################################
    def RLC(f, Kex, Ux, Uy, xp, yp, tgef, Dx, Dy):
        x1c, x2c = -Dx/2, Dx/2
        y1c, y2c = -Dy/2, Dy/2
        Ilm = spi.quad(partial(R2, Ux = Ux, x1c = x1c), x1c, x2c)[0]

        alpha = h * (R(xp, Ux, x1c) * np.cos(Uy * (yp-y1c)) * np.sinc(Ux * Dxp / (2*np.pi)) * np.sinc(Uy * Dxp / (2*np.pi))) * \
            (R(xp, Ux, x1c) * np.cos(Uy * (yp-y1c)) * np.sinc(Ux * Dxp / (2*np.pi)) * np.sinc(Uy * Dxp / (2*np.pi))) / ((y2c - y1c) * es * Ilm)
        if Uy != 0:
            alpha = 2 * alpha

        RLM = alpha/(2 * np.pi * f * tgef)
        CLM = 1/alpha
        LLM = alpha/(2 * np.pi * Kex / (2*np.pi*np.sqrt(u0*es)))**2
        return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM)))

    def Z(f, k01, k10, xp, yp, tgef01, tgef10, Dx, Dy):
        Ux01, Uy01 = 0, np.pi/Dy
        Ux10, Uy10 = np.pi/Dx, 0
        k = (2 * np.pi * f) * np.sqrt(es * u0)
        eta = np.sqrt(u0 / es)
        Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
        return RLC(f, k01, Ux01, Uy01, xp, yp, tgef01, Dx, Dy) + RLC(f, k10, Ux10, Uy10, xp, yp, tgef10, Dx, Dy) + 1j*Xp*Yp

    def S(Kdes, Dx, Dy):
        Dxa, Dya = Dx - 2*deltax, Dy - 2*deltay
        return (2 * h * Dya) / (2 * h * Dxa) + 0j

    def Probe(k01, k10, Kef01, Kef10, Dx, Dy):
        U10, U01 = np.pi/Dx, np.pi/Dy
        x1c, x2c = -Dx/2, Dx/2
        y1c = -Dy/2
        x1, x2 = x1c + deltax, x2c - deltax

        I01 = spi.quad(partial(R2, Ux = 0, x1c = x1c), x1c, x2c)[0]
        I10 = spi.quad(partial(R2, Ux = U10, x1c = x1c), x1c, x2c)[0]

        xp = np.linspace(x1, x2, 101)

        V = (I01 * R(x1c, U10, x1c))/(2 * I10 * R((x1c+x2c)/2, 0, x1c)) * (Kef01**2 - k01**2)/(Kef10**2 - k10**2) * S(np.real(Kef01), Dx, Dy)
        yp = y1c + np.arccos(R(xp, U10, x1c) / R(xp, 0, x1c) * np.abs(V)) / U01

        mask = ~np.isnan(yp)
        yp = yp[mask]
        xp = xp[mask]

        return yp, xp

    # Start of the design
    def imZin(p, Analysis = 0):
        k01 = k10 = 2*np.pi*flm_des*np.sqrt(u0*es)
        Dx = Dx_find(k10)
        Dy = Dy_find(k01)

        epsilon = 1
        tol = 1e-4
        steps = 0
        while epsilon > tol:
            steps += 1
            kl = k10 + p*(k01 - k10)
            k01ll = kl*tg01/2
            k10ll = kl*tg10/2
            phaseK = np.pi/2 - np.angle(S(kl, Dx, Dy))

            k01prev = k01
            k10prev = k10

            # Supose k01 > k10
            k10 = kl - ((k10ll+p*(k01ll-k10ll)) - np.sqrt((k10ll-p*(k01ll+k10ll))**2 - 4*k01ll*k10ll*(p-1)*p/np.cos(phaseK)**2) ) / (2*np.tan(phaseK)*(p-1))
            k01 = kl - ((k10ll+p*(k01ll-k10ll)) - np.sqrt((k10ll-p*(k01ll+k10ll))**2 - 4*k01ll*k10ll*(p-1)*p/np.cos(phaseK)**2) ) / (2*np.tan(phaseK)*p)
            if k10 > k01:
                k10 = kl - ((k10ll+p*(k01ll-k10ll)) + np.sqrt((k10ll-p*(k01ll+k10ll))**2 - 4*k01ll*k10ll*(p-1)*p/np.cos(phaseK)**2) ) / (2*np.tan(phaseK)*(p-1))
                k01 = kl - ((k10ll+p*(k01ll-k10ll)) + np.sqrt((k10ll-p*(k01ll+k10ll))**2 - 4*k01ll*k10ll*(p-1)*p/np.cos(phaseK)**2) ) / (2*np.tan(phaseK)*p)

            Dx = Dx_find(k10)
            Dy = Dy_find(k01)

            epsilon = np.max([np.abs(k01prev - k01), np.abs(k10prev - k10)])
            if steps > 50 or k10 is np.nan or k01 is np.nan:
                    return np.nan

        Py, Px = Probe(k01, k10, kl-1j*k01ll, kl-1j*k10ll, Dx, Dy)

        Pyx = interp1d(Px, Py)

        def Z50(x):
            return np.real(Z(flm_des, k01, k10, x, Pyx(x), tg01, tg10, Dx, Dy))-50
        # See the note about the bracket in synth_ant
        root = root_scalar(Z50, bracket=[Px[-1]*1e-3, Px[-1]], method='bisect')

        xp = np.array(root.root)

        Zin = Z(flm_des, k01, k10, xp, Pyx(xp), tg01, tg10, Dx, Dy)

        if  Analysis:
            Dya = Dy - 2*deltay
            Dxa = Dx - 2*deltax
            return Dxa, Dya, xp, Pyx(xp)
        return np.imag(Zin)

    x = np.round(np.linspace(0.05, 0.95, 10), 2)
    y_list = []
    for xi in x:
        warnings.simplefilter("ignore")
        try:
            y_list.append(imZin(xi, Analysis=0))
        except:
            y_list.append(np.nan)
    y = np.array(y_list, dtype=float)
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError("Not enough points to find a root.")
    if np.all(y>0) or np.all(y<0):
        raise ValueError("No interval found where the function changes sign.")

    idx = 0
    for i in range(len(y)-1):
        if y[i] * y[i+1] <= 0:
            idx = i
            if x[i+1]-x[i]<0.11:
                break
            else:
                raise ValueError("Too large an interval between points of sign change.")

    Wrapper = partial(imZin, Analysis = 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = root_scalar(Wrapper, bracket=[x[idx], x[idx+1]], method='bisect', xtol = 1e-4)
        Dxa, Dya, xp, yp = imZin(np.array(root.root), Analysis = 1)
    #print('p = ', np.array(root.root))

    parametros = np.array([Dxa, Dya, xp, yp])
    return parametros

# import matplotlib.pyplot as plt

# er = 9.8
# h = 0.003175
# center_freq = 1575.42e6
# geom = (er, h)

# band = np.round(((center_freq * er / 50 * h / 0.001524) / 1e6 + 60) / 1e3, 2) * 1e9

# # Síntese; se já tiver feito, reutilize os parametros e estimativas existentes.
# parametros, estimativas = synth_ant(center_freq, geom=geom)

# Dxa, Dya, xp, yp = parametros
# Dxp, Yp, tg01, tg10, deltax, deltay = estimativas

# sweep = (center_freq - band / 2, center_freq + band / 2, 2001)
# freq = np.linspace(sweep[0], sweep[1], sweep[2])
# print(f"Design parameters: Dxa = {Dxa*1000:.4f} mm, Dya = {Dya*1000:.4f} mm, xp = {xp*1000:.4f} mm, yp = {yp*1000:.4f} mm")
# Z_sweep = Z_par_g(
#     Dxp, Yp, tg01, tg10, deltax, deltay,
#     Dxa, Dya, xp, yp, sweep, geom=geom
# )

# fig, (ax_re, ax_im) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
# ax_re.plot(freq / 1e9, Z_sweep.real)
# ax_re.set_ylabel("Re{Z} (Ω)")
# ax_re.grid(True)

# ax_im.plot(freq / 1e9, Z_sweep.imag)
# ax_im.set_xlabel("Frequência (GHz)")
# ax_im.set_ylabel("Im{Z} (Ω)")
# ax_im.grid(True)

# fig.tight_layout()
# plt.show()