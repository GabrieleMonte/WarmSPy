import datetime
import numpy as np
from scipy.optimize import curve_fit, root
from scipy.interpolate import interp1d
from tqdm import trange
from scipy.integrate import odeint


def logspace(xini, xfin, ntot):
    log_array = 10.0**np.arange(xini, xfin +
                                (xfin - xini)/(ntot-1), (xfin - xini)/(ntot-1))
    if int(log_array[-1]) > int(10**xfin):
        return log_array[:-1]
    return log_array


class InflatonModel:
    def __init__(self, type, args, g, a1, Mpl):
        self.type = type
        self.args = args
        self.g = g
        self.a1 = a1
        self.Mpl = Mpl

    def Potential(self, ph):
        type = self.type
        args = self.args
        Mpl= self.Mpl
        if type == 'monomial':
            self.l = args[0]
            self.n = args[1]
            return self.l/np.math.factorial(self.n)*ph**self.n
        elif type == 'minimal':
            self.M = args[0]
            self.l = args[1]
            self.m = args[2]
            return self.M**4/self.l + self.m**2*ph**2/2
        elif type == 'beta_exponential':
            self.V0 = args[0]
            self.l = args[1]
            self.beta = args[2]
            return self.V0*(1-self.l*self.beta*ph/Mpl)**(1/self.beta)
        elif type == 'runaway':
            self.V0 = args[0]
            self.alpha = args[1]
            self.n= args[2]
            return self.V0*np.exp(-self.alpha*(ph/Mpl)**self.n)
        else:
            raise Exception('Potential type not recognized!')
    # derivative of the potential w.r.t. phi:

    def dPotential(self, ph):
        type = self.type
        args = self.args
        Mpl= self.Mpl
        if type == 'monomial':
            self.l = args[0]
            self.n = args[1]
            return self.l/np.math.factorial(self.n-1)*ph**(self.n-1)
        elif type == 'minimal':
            self.M = args[0]
            self.l = args[1]
            self.m = args[2]
            return self.m**2*ph
        elif type == 'beta_exponential':
            self.V0 = args[0]
            self.l = args[1]
            self.beta = args[2]
            return -self.V0*self.l/Mpl*(1-self.l*self.beta*ph/Mpl)**(1/self.beta-1)
        elif type == 'runaway':
            self.V0 = args[0]
            self.alpha = args[1]
            self.n= args[2]
            return -self.V0/Mpl*self.alpha*self.n*(ph/Mpl)**(self.n-1)*np.exp(-self.alpha*(ph/Mpl)**self.n)
        else:
            raise Exception('Potential type not recognized!')
    # second derivative of the potential w.r.t. phi:

    def d2Potential(self, ph):
        type = self.type
        args = self.args
        Mpl= self.Mpl
        if type == 'monomial':
            self.l = args[0]
            self.n = args[1]
            return self.l/np.math.factorial(self.n-2)*ph**(self.n-2)
        elif type == 'minimal':
            self.m = args[2]
            return self.m**2
        elif type == 'beta_exponential':
            self.V0 = args[0]
            self.l = args[1]
            self.beta = args[2]
            return self.V0*self.l**2/Mpl**2*(1-self.beta)*(1-self.l*self.beta*ph/Mpl)**(1/self.beta-2)
        elif type == 'runaway':
            self.V0 = args[0]
            self.alpha = args[1]
            self.n= args[2]
            return self.V0*(self.alpha*self.n/Mpl*(ph/Mpl)**(self.n-1))**2*np.exp(-self.alpha*(ph/Mpl)**self.n)-self.alpha*self.n*(self.n-1)/(Mpl**2)*(ph/Mpl)**(self.n-2)*self.V0*np.exp(-self.alpha*(ph/Mpl)**self.n)
        else:
            raise Exception('Potential type not recognized!')

    def rho_R(self, T):
        a1 = self.a1
        return a1*T**4

    def pr_R(self, T):
        return self.rho_R(T)/3

    def T_r(self, rhoR):
        a1 = self.a1
        return (rhoR/a1)**(1/4)

    def rho_phi(self, ph, dotph):
        return 1/2*dotph**2+self.Potential(ph)

    def pr_phi(self, ph, dotph):
        return 1/2*dotph**2-self.Potential(ph)

    def Rpowerspec(self, delphiv, delrhorv, Psirv, varphiv, H_bgv, T_bgv, dotphi_bgv):
        hatRphi = -varphiv+H_bgv/(dotphi_bgv)*delphiv
        hatRrho = -varphiv-H_bgv/(4/3*self.rho_R(T_bgv))*Psirv
        return (dotphi_bgv**2*hatRphi+4/3*self.rho_R(T_bgv)*hatRrho)/(dotphi_bgv**2+4/3*self.rho_R(T_bgv))
    # Hubble parameter and its derivative w.r.t. cosmic time t: ##

    def H(self, ph, dotph, rhoR):  # dotph is the derivative w.r.t. t
        V = self.Potential
        Mpl = self.Mpl
        return np.sqrt((V(ph)+rhoR+dotph**2/2)/(3*Mpl**2))

    def dotH(self, ph, dotph, rhoR):
        Mpl = self.Mpl
        return -1/(2*Mpl**2)*(dotph**2+4/3*rhoR)

    def ddotH(self, ph, dotph, rhoR, Q):
        H = self.H
        Mpl = self.Mpl
        dVdph = self.dPotential
        return 1/(Mpl**2)*(H(ph, dotph, rhoR)*(3+Q)*dotph**2+8/3*rhoR*H(ph, dotph, rhoR)+dotph*dVdph(ph))

    def Htau(self, ph, dotph, rhoR):  # dotph is the derivative w.r.t. \tau
        V = self.Potential
        Mpl = self.Mpl
        return np.sqrt((V(ph)+rhoR)/(3*Mpl**2-dotph**2/2))


def dW(delta_t, shape):
    current_date = datetime.datetime.now()
    seed_number = int(current_date.strftime("%S%f"))
    np.random.seed(seed_number)
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t), size=shape)


def Gamma_eff(Hv, Qv, Tv):
    return np.sqrt(6*Hv*Qv*Tv)


def Xi_q(Hv, Qv, Tv):
    nk_exp = np.exp(-Hv/Tv)
    nBE = nk_exp/(1-nk_exp)
    return np.sqrt((1+2*nBE)/np.pi**(3/2))*Hv*(9+12*np.pi*Qv)**(1/4)


class Background:
    def __init__(self, model, ph0, Q, verbose=False):
        self.model = model
        self.ph0 = ph0
        self.Q = Q
        self.verbose = verbose

    def dotphConstraint(self, dotph, ph, Q):
        V = self.model.Potential
        dVdph = self.model.dPotential
        Mpl = self.model.Mpl
        return 3*(1+Q)*dotph*np.sqrt((V(ph)+(3*Q+2)*dotph**2/4)/(3*Mpl**2))+dVdph(ph)

    def ICs_calculator(self, ph, Q):
        H = self.model.H
        sol = root(self.dotphConstraint, x0=[-10**(-10)], args=(ph, Q))
        [dotphv] = sol.x
        rhoR0 = 3/4*Q*dotphv**2
        H0 = H(ph, dotphv, 3/4*Q*dotphv**2)
        return dotphv, rhoR0, H0

    def diff_eq_tau(self, y, x, Q):
        dVdph = self.model.dPotential
        Htau = self.model.Htau
        y1, y2, y3, dy1dx = y  # y1=\phi, y2=\rho_R, y3= Ne and #dy1/dx=d\phi/dtau
        eq_y1 = -3*(1+Q)*dy1dx-dVdph(y1)/(Htau(y1, dy1dx, y2))**2
        eq_y2 = -4*y2+3*(Htau(y1, dy1dx, y2))**2*Q*dy1dx**2
        dydx = [dy1dx, eq_y2, 1, eq_y1]  # example differential equation
        return dydx

    def Compute_cosmo_pars_tau(self, y, Q, test=False):
        dVdph = self.model.dPotential
        Htau = self.model.Htau
        dotH = self.model.dotH
        ddotH = self.model.ddotH
        ph_vals = y[:, 0]
        rhoR_vals = y[:, 1]
        H_vals = Htau(y[:, 0], y[:, 3], y[:, 1])
        dotph_vals = y[:, 3]*H_vals
        eH_vals = -dotH(y[:, 0], dotph_vals, y[:, 1])/H_vals**2
        etaH_vals = ddotH(y[:, 0], dotph_vals, y[:, 1], Q) / \
            (dotH(y[:, 0], dotph_vals, y[:, 1])*H_vals)+2*eH_vals
        ddotph_vals = -3*H_vals*(1+Q)*dotph_vals**2-dVdph(y[:, 0])
        Ne_vals = y[:, 2]
        if test:
            return eH_vals, Ne_vals
        else:
            return ph_vals, dotph_vals, ddotph_vals, rhoR_vals, H_vals, eH_vals, etaH_vals, Ne_vals

    # Solving the background equations:
    def Bg_solver_tau(self, xmax, xlen, tauvals, Neinflation, tolerance=10**(-3)):
        verbose = self.verbose
        ph0 = self.ph0
        Q = self.Q
        dotph0, rhoR0, H0 = self.ICs_calculator(ph0, Q)
        y_ic = [ph0, rhoR0, 0, dotph0/H0]
        model = self.model
        xs = np.linspace(0, xmax, xlen)
        y = odeint(self.diff_eq_tau, y_ic, xs, args=(Q,))
        # If y has NaN values, I want to return an error and exit the code:
        if np.isnan(y).any():
            print(
                'Error: the solution has NaN values, make sure the initial conditions are correct first!')
            return
        phis, dotphis, ddotphis, rhoRs, Hs, eHs, etaHs, Nes = self.Compute_cosmo_pars_tau(
            y, Q)
        # If the max value of eH is less than 1, I want to return an error and exit the code:
        if max(eHs) < (1-tolerance):
            print(max(eHs))
            print(
                'Error: the solution has eH<1, make sure the initial conditions are correct first!')
            return
        eH1_index = min(range(len(eHs)),
                        key=lambda i: abs(eHs[i]-(1-tolerance)))
        delta_tau = Nes[eH1_index]-Neinflation
        taus = -(xs-delta_tau)
        phi_func = interp1d(taus, phis)
        dotphi_func = interp1d(taus, dotphis)
        ddotphi_func = interp1d(taus, ddotphis)
        Tr_func = interp1d(taus, model.T_r(rhoRs))
        H_func = interp1d(taus, Hs)
        eH_func = interp1d(taus, eHs)
        etaH_func = interp1d(taus, etaHs)
        if verbose:
            print('Bg computed!')
        return [phi_func(tauvals), dotphi_func(tauvals), ddotphi_func(tauvals), eH_func(tauvals), etaH_func(tauvals), H_func(tauvals), Tr_func(tauvals)]

    # Testing the intial conditions by iteratively solving the background equations:
    def Bg_solver_test(self, xmax, xlen, Neinflation, tolerance=10**(-3), learning_rate=0.03, max_iter=500, verbose=False):
        if xmax-Neinflation < 6:
            print(
                'Error: xmax-Neinflation should be larger than 6, otherwise the solution will not have enough e-folds for the perturbations to evolve!')
            return

        ph0 = self.ph0
        ph0_or = ph0  # Original value of ph0
        model = self.model
        V = model.Potential
        dVdph = model.dPotential
        Q = self.Q
        dotph0, rhoR0, H0 = self.ICs_calculator(ph0, Q)
        y_ic = [ph0, rhoR0, 0, dotph0/H0]
        xs = np.linspace(0, xmax, xlen)
        i = 0
        while i in range(max_iter):
            if ph0 < 0:
                # When ph0<0, we decrease the learning rate and reset ph0 to its original value:
                learning_rate = learning_rate/2
                if verbose:
                    print(
                        'phi0 is negative, decreasing the learning rate and trying again!')
                ph0 = ph0_or
                dotph0, rhoR0, H0 = self.ICs_calculator(ph0, Q)
                y_ic = [ph0, rhoR0, 0, dotph0/H0]
                i += 1
            if verbose:
                print(ph0)
            y = odeint(self.diff_eq_tau, y_ic, xs, args=(Q,))
            if np.isnan(y).any() == True:
                print(
                    'phi0 is too small, please increase its original value or modify the learning rate!')
                return
            else:
                eHs, Nes = self.Compute_cosmo_pars_tau(y, Q, True)
                eHmax_ind = min(range(len(eHs)), key=lambda i: abs(eHs[i]-1))
                if verbose:
                    print('ciaooooo')
                    print(np.max(eHs))
                if np.max(eHs) < (1-tolerance):
                    ph0 += (V(ph0)/dVdph(ph0))*(np.max(eHs)-1) / \
                        np.max(eHs)*learning_rate
                    # if ph0>0, reset the learning rate to its original value:
                    if ph0 > 0 and learning_rate < 0.03:
                        learning_rate = learning_rate*2
                    dotph0, rhoR0, H0 = self.ICs_calculator(ph0, Q)
                    y_ic = [ph0, rhoR0, 0, dotph0/H0]
                    if verbose:
                        print(ph0)
                    i += 1
                if (np.max(eHs) >= (1-tolerance)) and (int(Nes[eHmax_ind]) < (Neinflation+8)):
                    ph0 += (V(ph0)/dVdph(ph0))*(np.max(eHs)-1) / \
                        np.max(eHs)*learning_rate
                    dotph0, rhoR0, H0 = self.ICs_calculator(ph0, Q)
                    y_ic = [ph0, rhoR0, 0, dotph0/H0]
                    # print(ph0)
                    i += 1
                if (np.max(eHs) >= (1-tolerance)) and (int(Nes[eHmax_ind]) >= (Neinflation+8)):
                    break
            if i == max_iter-1:
                print('Error: the solution has not converged, try to either: (1) start from different initial conditions, (2) increase the number of max iterations, (3) modify the learning rate.')
                return
        if verbose:
            print(int(Nes[eHmax_ind]))
            print(eHs[eHmax_ind])
            print(ph0)
        return ph0

    def analytic_power_spectrum(self, xmax, xlen, tauvals, Neinflation, tolerance=10**(-3)):
        Q = self.Q
        Bg_solver_tau = self.Bg_solver_tau
        Bg_vars = Bg_solver_tau(xmax, xlen, tauvals, Neinflation, tolerance)
        dotphi_bg = Bg_vars[1]
        H_bg = Bg_vars[5]
        T_bg = Bg_vars[6]
        # Keep only the values of the background variables where tauvals=0:
        index_hor = np.argmin(np.abs(tauvals))
        dotphi_bg_hor = dotphi_bg[index_hor]
        H_bg_hor = H_bg[index_hor]
        T_bg_hor = T_bg[index_hor]
        # Compute the power spectrum:
        DeltaR2 = (H_bg_hor**2/(2*np.pi*dotphi_bg_hor))**2*(1+2/(np.exp(H_bg_hor /
                                                                        T_bg_hor)-1)+(2*np.sqrt(3)*np.pi*Q)/np.sqrt(3+4*np.pi*Q)*T_bg_hor/H_bg_hor)
        return DeltaR2


class Perturbations:
    def __init__(self, model, N, Nruns, tauvals, dtau, c, m, Q, bg_file, verbose=False, metric_perturbations=False, negligible_epsH_etaH=True):
        self.N = N
        self.Nruns = Nruns
        self.model = model
        self.tauvals = tauvals
        self.dtau = dtau
        self.c = c
        self.m = m
        self.Q = Q
        self.bg_file = bg_file
        self.phi_bg, self.dotphi_bg, self.ddotphi_bg, self.epsH_bg, self.etaH_bg, self.H_bg, self.T_bg = bg_file
        self.metric_perturbations = metric_perturbations
        self.negligible_epsH_etaH = negligible_epsH_etaH
        ##############################
        ###### Initial conditions#####
        ##############################
        self.delphi_ini = Gamma_eff(self.H_bg[0], self.Q, self.T_bg[0])
        self.ddelphidq_ini = 0
        self.delrhor_ini = 0
        self.Psir_ini = 0
        self.varphi_ini = 0
        self.IC = [self.delphi_ini, self.ddelphidq_ini,
                   self.delrhor_ini, self.Psir_ini, self.varphi_ini]
        #############################################################

    def EOMs(self, p_ics, b_ics, tau, c, m, Q):
        model = self.model
        metric_perturbations = self.metric_perturbations
        negligible_epsH_etaH = self.negligible_epsH_etaH
        Mpl = model.Mpl
        delphi0, ddelphidq0, delrhor0, Psir0, varphi0 = p_ics
        phv0, dotphv0, ddotphv0, epsHv0, etaHv0, Hv0, Tv0 = b_ics
        exp2tau = np.exp(2*tau)  # corresponds to z^2
        if negligible_epsH_etaH == True:
            epsHv0 = np.zeros(len(epsHv0))
            etaHv0 = np.zeros(len(etaHv0))
        d2delphidq2 = (1/(1-epsHv0)**(2))*(-epsHv0*(1+etaHv0-epsHv0)*ddelphidq0+3*(1+Q)*(1-epsHv0)*ddelphidq0-(exp2tau+model.d2Potential(phv0)/Hv0**2-3*m*Q*dotphv0/(phv0*Hv0))
                                           * delphi0-c/(Hv0*dotphv0)*delrhor0+(2*ddotphv0/Hv0**2-3/Hv0*(1+Q)*dotphv0)*varphi0+(dotphv0/Hv0**2)*(2/(Mpl)**2*(dotphv0*delphi0-Psir0)+Hv0*varphi0))
        ddelrhordq = 1/(1-epsHv0)*((4-3*c*Q*dotphv0**2/(4*model.rho_R(Tv0)))*delrhor0-Hv0*exp2tau*Psir0+6*Q*Hv0*dotphv0*(1-epsHv0)*ddelphidq0 +
                                   3*m*Q*dotphv0**2/phv0*delphi0-2*model.rho_R(Tv0)/(2*(Mpl)**2*Hv0)*(dotphv0*delphi0-Psir0)-3*(Q*dotphv0**2+4*model.rho_R(Tv0)/3)*varphi0)
        dPsirdq = 1/(1-epsHv0)*(3*Psir0+3*Q*dotphv0*delphi0 +
                                delrhor0/(3*Hv0)-4*model.rho_R(Tv0)*varphi0/(3*Hv0))
        if metric_perturbations:
            dvarphidq = 1/(1-epsHv0)*(varphi0+1/(2*(Mpl**2)*Hv0)
                                      * (dotphv0*delphi0-Psir0))
        else:
            dvarphidq = 0
        return ddelphidq0, d2delphidq2, ddelrhordq, dPsirdq, dvarphidq

    def solve(self, val, verbose=False):
        tauvals = self.tauvals
        dtau = self.dtau
        model = self.model
        c = self.c
        m = self.m
        Q = self.Q
        N = self.N
        Nruns = self.Nruns
        phi_bg = self.phi_bg
        dotphi_bg = self.dotphi_bg
        ddotphi_bg = self.ddotphi_bg
        epsH_bg = self.epsH_bg
        etaH_bg = self.etaH_bg
        H_bg = self.H_bg
        T_bg = self.T_bg
        IC = self.IC
        hatR2 = np.zeros((Nruns, N))
        Noise_th = Gamma_eff(H_bg, Q, T_bg) * \
            np.exp(3*tauvals/2)*dW(-dtau, (Nruns, N))
        Noise_qu = Xi_q(H_bg, Q, T_bg)*np.exp(3*tauvals/2) * \
            dW(-dtau, (Nruns, N))
        if verbose == True:
            print('Computed the noise terms, now solving the perturbation equations ...')
        for j in trange(Nruns):
            delphisoln = np.zeros(N)  # \delta\hat{\phi}
            ddelphidqsoln = np.zeros(N)  # d{\delta\hat{\phi}}/d\tau
            delrhorsoln = np.zeros(N)  # \delta\hat{\rho}_r
            Psirsoln = np.zeros(N)  # \hat{\Psi}_r
            varphisoln = np.zeros(N)  # \hat{\varphi}
            delphisoln[0], ddelphidqsoln[0], delrhorsoln[0], Psirsoln[0], varphisoln[0] = IC
            newIC = IC
            for i in range(N-1):
                bgICs = [phi_bg[i], dotphi_bg[i], ddotphi_bg[i],
                         epsH_bg[i], etaH_bg[i], H_bg[i], T_bg[i]]
                derivs = self.EOMs(newIC, bgICs, tauvals[i], c, m, Q)
                ddelphi = derivs[0] * dtau
                ddelphidq = derivs[1] * dtau
                ddelrhor = derivs[2] * dtau
                dPsir = derivs[3] * dtau
                dvarphi = derivs[4] * dtau
                delphisoln[i+1] = delphisoln[i] + ddelphi
                ddelphidqsoln[i+1] = ddelphidqsoln[i] + \
                    ddelphidq + Noise_th[j][i] + Noise_qu[j][i]
                delrhorsoln[i+1] = delrhorsoln[i] + ddelrhor
                Psirsoln[i+1] = Psirsoln[i] + dPsir
                varphisoln[i+1] = varphisoln[i] + dvarphi
                newIC = [delphisoln[i+1], ddelphidqsoln[i+1],
                         delrhorsoln[i+1], Psirsoln[i+1], varphisoln[i+1]]
            hatR2[j] = model.Rpowerspec(
                delphisoln, delrhorsoln, Psirsoln, varphisoln, H_bg, T_bg, dotphi_bg)**2
            del delphisoln, ddelphidqsoln, delrhorsoln, Psirsoln, newIC,
        hatR2avg = np.mean(hatR2, axis=0)
        hatR2std = np.std(hatR2, axis=0)
        # I want the value of hatR2avg that corresponds to the index for which tauvals=0:
        index_hor = np.argmin(np.abs(tauvals))
        if verbose:
            print(index_hor)
        hatR2avg_hor = 1/(2*np.pi**2)*hatR2avg[index_hor]
        hatR2std_hor = 1/(2*np.pi**2)*hatR2std[index_hor]
        del Noise_qu, Noise_th
        return hatR2avg_hor, hatR2std_hor

# Fitting the growth factor function given by the perturbation data:


class Growth_factor:
    def __init__(self, model, Qvals, ph0s, hatR2avg, hatR2std, Nruns, xmax, xlen, tauvals, Neinflation, c, m):
        self.Qvals = Qvals
        self.ph0s = ph0s
        self.hatR2avg = hatR2avg
        self.hatR2std = hatR2std
        self.Nruns = Nruns
        self.model = model
        self.c = c
        self.m = m
        self.xmax = xmax
        self.xlen = xlen
        self.tauvals = tauvals
        self.Neinflation = Neinflation

    def growth_factor(self, analytic_power_spectrum=False,deltaR2_analytic =[]):
        Nruns = self.Nruns
        hatR2avg = self.hatR2avg
        hatR2std = self.hatR2std
        model = self.model
        Qvals = self.Qvals
        ph0s = self.ph0s
        xmax = self.xmax
        xlen = self.xlen
        tauvals = self.tauvals
        Neinflation = self.Neinflation
        if analytic_power_spectrum:
            DeltaR2_analytic = deltaR2_analytic
        else:
            DeltaR2_analytic = np.zeros(len(Qvals))
            # Computing the analytic power spectrum:
            for i in trange(len(Qvals)):
                Bg = Background(model, ph0s[i], Qvals[i])
                DeltaR2_analytic[i] = Bg.analytic_power_spectrum(xmax, xlen, tauvals, Neinflation)
            print('Analytic power spectrum computed!')
        # Computing the growth factor, its signal and noise:
        growth_factor_signal = hatR2avg/DeltaR2_analytic
        growth_factor_noise = hatR2std/DeltaR2_analytic*1/np.sqrt(Nruns)
        return growth_factor_signal, growth_factor_noise
    # Complicated function to fit the growth factor for a postive c:

    def growth_factor_fit_func_positive_c_complex(self, x, exp1, exp2, exp3, exp4, exp5, exp6, a1, a2, a3, b1, b2, b3):
        c = self.c
        return ((1+np.exp(a1)*x**exp1)/(1+np.exp(a2)*x**exp2))**c+np.exp(a3)*x**(exp3)*(1+b1*x**(exp4))/(1+b2*x**(exp5))+b3*x**(exp6)
    # Simple function to fit the growth factor for a positive c:

    def growth_factor_fit_func_positive_c_simple(self, x, exp1, exp2, a1, a2):
        return 1+a1*x**(exp1)+a2*x**(exp2)
    # Function to fit the growth factor for a negative c:

    def growth_factor_fit_func_negative_c(self, x, exp1, exp2, a1, a2, a3, a4):
        return (1+a1*x**exp1)**a2/(1+a3*x**exp2)**a4

    def growth_factor_fit(self, method='complex', make_your_own_bounds=False, lower_bounds=0, upper_bounds=0, output_data=False, analytic_power_spectrum=False, deltaR2_analytic=[]):
        global popt, pcov, low_bounds, up_bounds
        Qvals = self.Qvals
        c = self.c
        growth_factor_signal, growth_factor_noise = self.growth_factor(analytic_power_spectrum, deltaR2_analytic)
        if c < 0:
            if make_your_own_bounds:
                low_bounds = lower_bounds
                up_bounds = upper_bounds
            else:
                low_bounds = [0.1, 0.1, 0, 0, 0.1, 0.1]
                up_bounds = [5, 5, 5, 5, 5, 5]
            popt, pcov = curve_fit(self.growth_factor_fit_func_negative_c, Qvals,
                                   growth_factor_signal, sigma=growth_factor_noise, bounds=(low_bounds, up_bounds))
        if method == 'complex' and c > 0:
            if make_your_own_bounds:
                low_bounds = lower_bounds
                up_bounds = upper_bounds
            else:
                low_bounds = [0.01, 0.01, 0.01, 0.01,
                              0.01, 0.01, 0, 0, 0, 0, 0, 0]
                up_bounds = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
            popt, pcov = curve_fit(self.growth_factor_fit_func_positive_c_complex, Qvals,
                                   growth_factor_signal, sigma=growth_factor_noise, bounds=(low_bounds, up_bounds))
        if method == 'simple' and c > 0:
            if make_your_own_bounds:
                low_bounds = lower_bounds
                up_bounds = upper_bounds
            else:
                low_bounds = [0.1, 0.1, 0, 0]
                up_bounds = [10, 10, 10, 10]
            popt, pcov = curve_fit(self.growth_factor_fit_func_positive_c_simple, Qvals,
                                   growth_factor_signal, sigma=growth_factor_noise, bounds=(low_bounds, up_bounds))
        perr = np.sqrt(np.diag(pcov))
        if output_data:
            return popt, perr, growth_factor_signal, growth_factor_noise
        else:
            return popt, perr
