import datetime
import os
import multiprocessing as mp
import numpy as np
from scipy.optimize import curve_fit, root
from scipy.interpolate import interp1d
from tqdm import trange
from scipy.integrate import odeint

class InflatonModel:
    def __init__(self, type, args, g, a1, Mpl):
        self.type = type #type of Potential
        self.args = args #list of arguments of the potential
        self.g = g #Relativistic degrees of freedom
        self.a1 = a1 #\equiv \pi^2/30*g
        self.Mpl = Mpl #Planck mass

    def Potential(self, ph):
        type = self.type
        args = self.args
        Mpl= self.Mpl
        if type == 'monomial': #V(\phi)= l/n!*phi^n
            self.l = args[0]
            self.n = args[1]
            return self.l/np.math.factorial(self.n)*ph**self.n
        elif type == 'hilltop': #V(\phi)= V0*(1-|\gamma|/2n*(phi/Mpl)^2n)
            self.l=args[0]
            self.n=args[1]
            self.ph0=args[2]
            return self.l*Mpl**4/(2*self.n)*(1-(ph/self.ph0)**(2*self.n))**2
        elif type == 'minimal': #V(\phi)= M^4/l + m^2*\phi^2/2
            self.M = args[0]
            self.l = args[1]
            self.m = args[2]
            self.g_sigma=args[3] #The squared mass of the waterfall field is -M^2+g_sigma*phi^2
            return self.M**4/self.l + self.m**2*ph**2/2
        elif type=='natural': #V(\phi)= L^4*(1+cos(N*\phi/f))
            self.L=args[0]
            self.f=args[1]
            self.N=args[2]
            return self.L**4 *(1+np.cos(self.N* ph/self.f))
        elif type == 'beta_exponential': #V(\phi)= V0*(1-l*beta*\phi/Mpl)^1/beta
            self.V0 = args[0]
            self.l = args[1]
            self.beta = args[2]
            return self.V0*(1-self.l*self.beta*ph/Mpl)**(1/self.beta)
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
        elif type == 'hilltop':
            self.l=args[0]
            self.n=args[1]
            self.ph0=args[2]
            return -2*self.l*Mpl**4/self.ph0*(ph/self.ph0)**(2*self.n-1)*(1-(ph/self.ph0)**(2*self.n))
        elif type=='natural':
            self.L=args[0]
            self.f=args[1]
            self.N=args[2]
            return -self.L**4*self.N/self.f*np.sin(self.N* ph/self.f)
        elif type == 'beta_exponential':
            self.V0 = args[0]
            self.l = args[1]
            self.beta = args[2]
            return -self.V0*self.l/Mpl*(1-self.l*self.beta*ph/Mpl)**(1/self.beta-1)
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
        elif type == 'hilltop':
            self.l=args[0]
            self.n=args[1]
            self.ph0=args[2]
            return 2*self.l*Mpl**4*(-4*self.n*(2*self.n-1)/self.ph0**2*(ph/self.ph0)**(2*self.n-2)*(1-(ph/self.ph0)**(2*self.n))+8*self.n**2/self.ph0**2*(ph/self.ph0)**(4*self.n-2))
        elif type=='natural':
            self.L=args[0]
            self.f=args[1]
            self.N=args[2]
            return self.L**4*self.N**2/self.f**2*np.cos(self.N* ph/self.f)
        elif type == 'beta_exponential':
            self.V0 = args[0]
            self.l = args[1]
            self.beta = args[2]
            return self.V0*self.l**2/Mpl**2*(1-self.beta)*(1-self.l*self.beta*ph/Mpl)**(1/self.beta-2)
        else:
            raise Exception('Potential type not recognized!')

    def rho_R(self, T): #radiation density
        a1 = self.a1
        return a1*T**4

    def pr_R(self, T): #radiation pressure
        return self.rho_R(T)/3

    def T_r(self, rhoR): #temperature of the radiation bath
        a1 = self.a1
        return (rhoR/a1)**(1/4)

    def rho_phi(self, ph, dotph): #scalar field density
        return 1/2*dotph**2+self.Potential(ph)

    def pr_phi(self, ph, dotph): #scalar field pressure
        return 1/2*dotph**2-self.Potential(ph)


    def Rpowerspec(self, delphiv, delrhorv, Psirv, varphiv, H_bgv, T_bgv, dotphi_bgv): #power spectrum of the curvature perturbations
        hatRphi = -varphiv+H_bgv/(dotphi_bgv)*delphiv
        hatRrho = -varphiv-H_bgv/(4/3*self.rho_R(T_bgv))*Psirv
        return (dotphi_bgv**2*hatRphi+4/3*self.rho_R(T_bgv)*hatRrho)/(dotphi_bgv**2+4/3*self.rho_R(T_bgv))

    """Hubble parameter and its derivatives w.r.t. cosmic time t:
        Note: dotph is the derivative w.r.t. cosmic time t"""
    def H(self, ph, dotph, rhoR):
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

    #Hubble parameter w.r.t. the number of e-folds Ne:
    def H_Ne(self, ph, dotph, rhoR):  # dotph is the derivative w.r.t. Ne
        V = self.Potential
        Mpl = self.Mpl
        return np.sqrt((V(ph)+rhoR)/(3*Mpl**2-dotph**2/2))

    r"""The dissipation rate is define as: \Upsilon=C_\Upsilon*T^c/\phi^{m}.
        If we take Q=\Gamma/(3H) to not be constant, then we have to define C_\Upsilon,
        in terms of Q,c and m"""

    def C_Upsilon(self, ph,dotph,rhoR,Q, c, m):
        T_r = self.T_r
        H=self.H
        return 3*H(ph,dotph,rhoR)*Q*(ph**m)*T_r(rhoR)**(-c)

    r""""Evolution of Q as a function of C_Upislon, c and m:"""

    def Q_evolution(self,ph,dotph,rhoR,c,m,C_Upsilon):
        T_r = self.T_r
        H=self.H
        return C_Upsilon/(3*H(ph,dotph,rhoR))*T_r(rhoR)**c/ph**m
    #Q evolution w.r.t. the number of e-folds Ne:
    def Q_evolution_Ne(self, ph,dotph,rhoR,c,m,C_Upsilon):  # dotph is the derivative w.r.t. Ne
        H_Ne=self.H_Ne
        T_r = self.T_r
        return C_Upsilon/(3*H_Ne(ph,dotph,rhoR))*T_r(rhoR)**c/ph**m

class Background:
    def __init__(self, model, ph0, Q, verbose=False,Qconstant=True,Qvariables=[]):
        self.model = model
        self.ph0 = ph0
        self.Q = Q
        self.verbose = verbose
        self.Qconstant = Qconstant
        r"""If Q is not constant, then remember to set Qvariables=[c,m] where c and m are the power-law
         exponents of the dissipation rate respectively for the temperature and the inflaton field"""
        self.Qvariables=Qvariables
    """"These functions are used to calculate the initial conditions for the background evolution, for a given value of \phi_0 and Q."""

    def dotphConstraint(self, dotph, ph, Q): #constraint on the derivative of the scalar field w.r.t. cosmic time t
        V = self.model.Potential
        dVdph = self.model.dPotential
        Mpl = self.model.Mpl
        return 3*(1+Q)*dotph*np.sqrt((V(ph)+(3*Q+2)*dotph**2/4)/(3*Mpl**2))+dVdph(ph)

    def ICs_calculator(self, ph, Q): #calculates the initial conditions for the background evolution
        H = self.model.H
        sol = root(self.dotphConstraint, x0=[-10**(-10)*ph], args=(ph, Q))
        [dotph0] = sol.x
        rhoR0 = 3/4*Q*dotph0**2
        H0 = H(ph, dotph0, 3/4*Q*dotph0**2)
        return dotph0, rhoR0, H0

    def Phi0_slow_roll_guesser(self, Ne_inflation): #calculates the initial value of \phi for the background evolution, assuming slow-roll and "Ne_inflation"" e-folds of inflation
        Mpl = self.model.Mpl
        Q=self.Q
        potential_type = self.model.type
        potential_args = self.model.args
        if potential_type == 'monomial':
            n = potential_args[1]
            return Mpl*n/(np.sqrt(2*(1+Q)))*np.sqrt(1+4*Ne_inflation/n)
        elif potential_type == 'beta_exponential':
            l = potential_args[1]
            beta = potential_args[2]
            return Mpl/(l*beta)*(1-np.sqrt(l**2*(1+4*Ne_inflation*beta)/(2*(1+Q))))
        elif potential_type == 'natural':
            L= potential_args[0]
            f= potential_args[1]
            Nphi= potential_args[2]
            f2_eff= (f/(Mpl*Nphi))**2*(1+Q)
            ph_end=np.arccos((1-2*f2_eff)/(1+2*f2_eff))
            return 2*np.arcsin(np.sin(ph_end/2)*np.exp(-Ne_inflation/(2*f2_eff)))*f/Nphi

    def diff_eq_Ne(self, y, x, Q_pars): #differential equation for the background evolution w.r.t. the number of e-folds Ne
        dVdph = self.model.dPotential
        H_Ne = self.model.H_Ne
        Qconstant = self.Qconstant
        if Qconstant:
            Q=Q_pars
            y1, y2, y3, dy1dx = y  # y1=\phi, y2=\rho_R, y3= Ne and #dy1/dx=d\phi/dNe
            #if y2 is negative, the differential equation return an error and quits the integration:
            if y2 < 0:
                raise Exception('rho_R is negative')
            eq_y1 = -3*(1+Q)*dy1dx-dVdph(y1)/(H_Ne(y1, dy1dx, y2))**2
            eq_y2 = -4*y2+3*(H_Ne(y1, dy1dx, y2))**2*Q*dy1dx**2
            dydx = [dy1dx, eq_y2, 1, eq_y1]  # example differential equation
            return dydx
        else:
            c,m,C_Upsilon=Q_pars
            Q_evolution_Ne=self.model.Q_evolution_Ne
            y1, y2, y3, dy1dx = y #y1=\phi, y2=\rho_R, y3= Ne and #dy1/dx=d\phi/dtau
            eq_y1=-3*(1+Q_evolution_Ne(y1,dy1dx,y2,c,m,C_Upsilon))*dy1dx-dVdph(y1)/(H_Ne(y1,dy1dx,y2))**2
            eq_y2= -4*y2+3*(H_Ne(y1,dy1dx,y2))**2*Q_evolution_Ne(y1,dy1dx,y2,c,m,C_Upsilon)*dy1dx**2
            dydx = [dy1dx, eq_y2, 1, eq_y1] # example differential equation
            return dydx

    def Compute_cosmo_pars_Ne(self, y,test=False): #calculates the background parameters as a function of the number of e-folds Ne
        dVdph = self.model.dPotential
        H_Ne = self.model.H_Ne
        dotH = self.model.dotH
        ddotH = self.model.ddotH
        Qconstant = self.Qconstant
        ph_vals = y[:, 0]
        rhoR_vals = y[:, 1]
        H_vals = H_Ne(y[:, 0], y[:, 3], y[:, 1])
        dotph_vals = y[:, 3]*H_vals #dotph is the derivative w.r.t. t evaluated as a function of Ne
        epsH_vals = -dotH(y[:, 0], dotph_vals, y[:, 1])/H_vals**2
        Ne_vals = y[:, 2]
        if Qconstant:
            Q=self.Q
            etaH_vals = ddotH(y[:, 0], dotph_vals, y[:, 1], Q)/(dotH(y[:, 0], dotph_vals, y[:, 1])*H_vals)+2*epsH_vals
            ddotph_vals = -3*H_vals*(1+Q)*dotph_vals**2-dVdph(y[:, 0])
            Q_vals = Q*np.ones(len(y[:, 0]))
        else:
            c,m=self.Qvariables
            C_Upsilon=self.model.C_Upsilon(ph_vals[0],dotph_vals[0],rhoR_vals[0],self.Q,c,m)
            Q_evolution_Ne=self.model.Q_evolution_Ne
            Q_vals=Q_evolution_Ne(y[:, 0], y[:, 3], y[:, 1],c,m,C_Upsilon)
        eta_vals = ddotH(y[:, 0], dotph_vals, y[:, 1], Q_vals)/(dotH(y[:, 0], dotph_vals, y[:, 1])*H_vals)+2*epsH_vals
        ddotph_vals = -3*H_vals*(1+Q_vals)*dotph_vals**2-dVdph(y[:, 0])
        if test:
            return epsH_vals, Ne_vals
        else:
            return ph_vals, dotph_vals, ddotph_vals, rhoR_vals, H_vals, epsH_vals, eta_vals, Ne_vals, Q_vals

    # Solving the background equations:
    def Bg_solver_tau(self, Ne_max, Ne_len, tauvals, Ne_inflation, tolerance=10**(-3), rtol_value=1.0e-6,atol_value=1.0e-10):
        verbose = self.verbose
        ph0 = self.ph0
        Q = self.Q
        dotph0, rhoR0, H0 = self.ICs_calculator(ph0, Q)
        y_ic = [ph0, rhoR0, 0, dotph0/H0]
        model = self.model
        Qconstant = self.Qconstant
        Nes = np.linspace(0, Ne_max, Ne_len)
        if Qconstant:
            y = odeint(self.diff_eq_Ne, y_ic, Nes, args=(Q,),rtol=rtol_value,atol=atol_value)
            if np.isnan(y).any():
                print('Error: the solution has NaN values, make sure the initial conditions are correct first!')
                return
        else:
            c,m=self.Qvariables
            C_Upsilon = self.model.C_Upsilon(ph0,dotph0,rhoR0,Q,c,m)
            y = odeint(self.diff_eq_Ne, y_ic, Nes, args=([c,m,C_Upsilon],),rtol=rtol_value,atol=atol_value)
            # If y has NaN values, I want to return an error and exit the code:
            if np.isnan(y).any():
                print('Error: the solution has NaN values, make sure the initial conditions are correct first!')
                return
        phis, dotphis, ddotphis, rhoRs, Hs, epsHs, etaHs, Nes, Qs = self.Compute_cosmo_pars_Ne(y)
        if round(max(epsHs), 3) < (1-tolerance):
            print(max(epsHs))
            print(
                'Error: the solution has epsH<1, make sure the initial conditions are correct first!')
            return
        epsH1_index = min(range(len(epsHs)),
                        key=lambda i: abs(epsHs[i]-(1-tolerance))) #index of the first value for which eps_H>(1-tolerance), i.e. the first value for which inflation ends.
        delta_Ne = Nes[epsH1_index]-Ne_inflation #number of e-folds prior to the "Ne_inflation" e-folds of inflation
        taus = -(Nes-delta_Ne) #\tau\equiv \ln(k/(aH)) with \tau=0 that corresponds to Ne_inflation e-folds before the end of inflation
        phi_func = interp1d(taus, phis)
        dotphi_func = interp1d(taus, dotphis)
        ddotphi_func = interp1d(taus, ddotphis)
        Tr_func = interp1d(taus, model.T_r(rhoRs))
        H_func = interp1d(taus, Hs)
        epsH_func = interp1d(taus, epsHs)
        etaH_func = interp1d(taus, etaHs)
        Qs_func = interp1d(taus, Qs)
        if verbose:
            print('Bg computed!')
        return [phi_func(tauvals), dotphi_func(tauvals), ddotphi_func(tauvals), epsH_func(tauvals), etaH_func(tauvals), H_func(tauvals), Tr_func(tauvals), Qs_func(tauvals)]

    # Testing the intial conditions by iteratively solving the background equations:
    def Bg_solver_test(self, Ne_max, Ne_len, Ne_inflation, tolerance=1*10**(-3), learning_rate=0.03, max_iter=1000, verbose=False,rtol_value=1.0e-6,atol_value=1.0e-10):
        if Ne_max-Ne_inflation < 6:
            print(
                'Error: Ne_max-Ne_inflation should be larger than 6, otherwise the solution will not have enough e-folds for the perturbations to evolve!')
            return
        or_learning_rate = learning_rate
        ph0 = self.ph0
        ph0_or = ph0  # Original value of ph0
        model = self.model
        V = model.Potential
        dVdph = model.dPotential
        Q = self.Q
        Qconstant = self.Qconstant
        dotph0, rhoR0, H0 = self.ICs_calculator(ph0, Q)
        y_ic = [ph0, rhoR0, 0, dotph0/H0]
        if Qconstant:
            Q_pars=Q
        else:
            c,m=self.Qvariables
            C_Upsilon = self.model.C_Upsilon(ph0,dotph0,rhoR0,Q,c,m)
            Q_pars=[c,m,C_Upsilon]
        Nes = np.linspace(0, Ne_max, Ne_len)
        i = 0
        while i in range(max_iter):
            if verbose:
                print(r"$\phi_0$ = ", ph0)
            if ph0 < 0:
                # When ph0<0, we decrease the learning rate and reset ph0 to its original value:
                learning_rate = learning_rate/2
                if verbose:
                    print(
                        'phi0 is negative, decreasing the learning rate and trying again!')
                ph0 = ph0_or
                dotph0, rhoR0, H0 = self.ICs_calculator(ph0, Q)
                y_ic = [ph0, rhoR0, 0, dotph0/H0]
                if not Qconstant:
                    C_Upsilon = self.model.C_Upsilon(ph0,dotph0,rhoR0,Q,c,m)
                    Q_pars=[c,m,C_Upsilon]
                i += 1
            if ph0>0:
                y = odeint(self.diff_eq_Ne, y_ic, Nes, args=(Q_pars,),rtol=rtol_value,atol=atol_value)
                if np.isnan(y).any() == True:
                    print(
                    'phi0 is too small, please increase its original value or modify the learning rate!')
                    return
                else:
                    epsHs, Nes = self.Compute_cosmo_pars_Ne(y, True)
                    epsHmax_ind = min(range(len(epsHs)), key=lambda i: abs(epsHs[i]-1))
                    if verbose:
                        print(r"The maximum value of $\epsilon_H$ is: ", np.max(epsHs))
                    if (np.max(epsHs)-(1-tolerance)) <=0 :
                        ph0 +=(V(ph0)/dVdph(ph0))*(np.max(epsHs)-1) /np.max(epsHs)*learning_rate
                        
                        # if ph0>0, reset the learning rate to its original value:
                        if ph0 > 0 and learning_rate < or_learning_rate:
                            learning_rate = learning_rate*2
                        dotph0, rhoR0, H0 = self.ICs_calculator(ph0, Q)
                        y_ic = [ph0, rhoR0, 0, dotph0/H0]
                        if not Qconstant:
                            C_Upsilon = self.model.C_Upsilon(ph0,dotph0,rhoR0,Q,c,m)
                            Q_pars=[c,m,C_Upsilon]
                        i += 1
                    if (((np.max(epsHs)-(1-tolerance)) >0 ) and (int(Nes[epsHmax_ind]) < (Ne_max-1))) or (((np.max(epsHs)-1) > 0) and (int(Nes[epsHmax_ind]) >= (Ne_max-1))):
                        ph0 += (dVdph(ph0)/V(ph0))*(np.max(epsHs)-1) * np.max(epsHs)*learning_rate
                        dotph0, rhoR0, H0 = self.ICs_calculator(ph0, Q)
                        y_ic = [ph0, rhoR0, 0, dotph0/H0]
                        if not Qconstant:
                            C_Upsilon = self.model.C_Upsilon(ph0,dotph0,rhoR0,Q,c,m)
                            Q_pars=[c,m,C_Upsilon]
                        # print(ph0)
                        i += 1
                    if ((np.max(epsHs)-(1-tolerance)) >0 ) and ((np.max(epsHs)-1)<0) and (int(Nes[epsHmax_ind]) >= (Ne_max-1)):
                        break
            if i == max_iter-1:
                print('Error: the solution has not converged, try to either: (1) start from different initial conditions, (2) increase the number of max iterations, (3) modify the learning rate.')
                return
        if verbose:
            print()
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("The number of e-folds of inflation we get is: ", int(Nes[epsHmax_ind]))
            print("With a $\phi_0$ of: ", ph0)
            print("The maximum value of $\epsilon_H$ is: ", np.max(epsHs))
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print()
        return ph0

    def analytic_power_spectrum(self, Ne_max, Ne_len, tauvals, Ne_inflation, tolerance=10**(-3)): #Function to compute the analytic approximation of the power spectrum
        Qconstant = self.Qconstant
        Bg_solver_tau = self.Bg_solver_tau
        Bg_vars = Bg_solver_tau(Ne_max, Ne_len, tauvals, Ne_inflation, tolerance)
        dotphi_bg = Bg_vars[1]
        H_bg = Bg_vars[5]
        T_bg = Bg_vars[6]
        # Keep only the values of the background variables for which \tau=0, i.e. Ne_inflation e-folds before the end of inflation:
        index_hor = np.argmin(np.abs(tauvals))
        dotphi_bg_hor = dotphi_bg[index_hor]
        H_bg_hor = H_bg[index_hor]
        T_bg_hor = T_bg[index_hor]
        # Compute the power spectrum:
        if Qconstant:
            Q_bg_hor=self.Q
        else:
            Q_bg=Bg_vars[7]
            Q_bg_hor = Q_bg[index_hor]
        DeltaR2 = (H_bg_hor**2/(2*np.pi*dotphi_bg_hor))**2*(1+2/(np.exp(H_bg_hor /T_bg_hor)-1)+(2*np.sqrt(3)*np.pi*Q_bg_hor)/np.sqrt(3+4*np.pi*Q_bg_hor)*T_bg_hor/H_bg_hor)
        return DeltaR2

########################################################################################################################
"""These functions are used to generate the Gaussian noise terms that enter the perturbation equations."""

def dW(delta_t, shape):
    current_date = datetime.datetime.now()
    seed_number = int(current_date.strftime("%S%f"))
    np.random.seed(seed_number)
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t), size=shape)

#Amplitude of the Thermal Noise:
def Xi_T(H, Q, T):
    return np.sqrt(6*H*Q*T)

#Amplitude of the Quantum Noise:
def Xi_q(H, Q, T):
    nk_exp = np.exp(-H/T)
    nBE = nk_exp/(1-nk_exp)
    return np.sqrt((1+2*nBE)/np.pi)*H*(9+12*np.pi*Q)**(1/4)


########################################################################################################################

class Perturbations:
    def __init__(self, model, N, Nruns, tauvals, dtau, c, m,bg_file, verbose=False, metric_perturbations=False, negligible_epsH_etaH=True):
        self.N = N #length of the tau array where tau\equiv ln(k/(aH))
        self.Nruns = Nruns #Number of runs to average over the solutions to the stochastic differential equations caractherizing the perturbations
        self.model = model
        self.tauvals = tauvals #array of values of tau
        self.dtau = dtau #step size in tau
        #Recall: the dissipation rate \Gamma\equiv 3HQ \propto T^c/\phi^m
        self.c = c #power of T in the dissipation rate
        self.m = m #inverse power of phi in the dissipation rate
        self.bg_file = bg_file #file containing the background variables
        self.phi_bg, self.dotphi_bg, self.ddotphi_bg, self.epsH_bg, self.etaH_bg, self.H_bg, self.T_bg , self.Q_bg= bg_file
        self.metric_perturbations = metric_perturbations
        self.negligible_epsH_etaH = negligible_epsH_etaH
        ##############################
        ###### Initial conditions#####
        ##############################
        self.delphi_ini = Xi_T(self.H_bg[0], self.Q_bg[0], self.T_bg[0]) # \delta\hat{\phi}(0)
        self.ddelphidtau_ini = 0  # d{\delta\hat{\phi}}/d\tau (0)
        self.delrhor_ini = 0 # \delta\hat{\rho}_R(0)
        self.Psir_ini = 0 # \hat{\Psi}_R(0)
        self.varphi_ini = 0 # \hat{\varphi}(0)
        self.IC = [self.delphi_ini, self.ddelphidtau_ini,
                   self.delrhor_ini, self.Psir_ini, self.varphi_ini]
        #############################################################

    def EOMs(self, p_ics, b_ics, tau, c, m):
        model = self.model
        metric_perturbations = self.metric_perturbations
        negligible_epsH_etaH = self.negligible_epsH_etaH
        Mpl = model.Mpl
        delphi0, ddelphidtau0, delrhor0, Psir0, varphi0 = p_ics
        phv0, dotphv0, ddotphv0, epsHv0_val, etaHv0_val, Hv0, Tv0, Qv0 = b_ics
        epsHv0=0
        etaHv0=0
        exp2tau = np.exp(2*tau)  # corresponds to z^2
        if negligible_epsH_etaH == False:
            epsHv0 = epsHv0_val
            etaHv0 = etaHv0_val
        d2delphidtau2 = (1/(1-epsHv0)**(2))*(-epsHv0*(1+etaHv0-epsHv0)*ddelphidtau0+3*(1+Qv0)*(1-epsHv0)*ddelphidtau0-(exp2tau+model.d2Potential(phv0)/Hv0**2-3*m*Qv0*dotphv0/(phv0*Hv0))
                                           * delphi0-c/(Hv0*dotphv0)*delrhor0+(2*ddotphv0/Hv0**2-3/Hv0*(1+Qv0)*dotphv0)*varphi0+(dotphv0/Hv0**2)*(2/(Mpl)**2*(dotphv0*delphi0-Psir0)+Hv0*varphi0))
        ddelrhordtau = 1/(1-epsHv0)*((4-3*c*Qv0*dotphv0**2/(4*model.rho_R(Tv0)))*delrhor0-Hv0*exp2tau*Psir0+6*Qv0*Hv0*dotphv0*(1-epsHv0)*ddelphidtau0 +
                                   3*m*Qv0*dotphv0**2/phv0*delphi0-2*model.rho_R(Tv0)/(2*(Mpl)**2*Hv0)*(dotphv0*delphi0-Psir0)-3*(Qv0*dotphv0**2+4*model.rho_R(Tv0)/3)*varphi0)
        dPsirdtau = 1/(1-epsHv0)*(3*Psir0+3*Qv0*dotphv0*delphi0 +
                                delrhor0/(3*Hv0)-4*model.rho_R(Tv0)*varphi0/(3*Hv0))
        dvarphidtau = 0
        if metric_perturbations:
            dvarphidtau = 1/(1-epsHv0)*(varphi0+1/(2*(Mpl**2)*Hv0)
                                      * (dotphv0*delphi0-Psir0))
        return ddelphidtau0, d2delphidtau2, ddelrhordtau, dPsirdtau, dvarphidtau

    def solve(self, val, verbose=False):
        tauvals = self.tauvals
        dtau = self.dtau
        model = self.model
        c = self.c
        m = self.m
        N = self.N
        Nruns = self.Nruns
        phi_bg = self.phi_bg
        dotphi_bg = self.dotphi_bg
        ddotphi_bg = self.ddotphi_bg
        epsH_bg = self.epsH_bg
        etaH_bg = self.etaH_bg
        H_bg = self.H_bg
        T_bg = self.T_bg
        Q_bg=self.Q_bg
        IC = self.IC
        # Compute the noise terms:
        Noise_th = Xi_T(H_bg, Q_bg, T_bg) * \
            np.exp(3*tauvals/2)*dW(-dtau, N) #Thermal noise
        Noise_qu = Xi_q(H_bg, Q_bg, T_bg)*np.exp(3*tauvals/2) * \
            dW(-dtau, N) #Quantum noise
        if verbose == True:
            print('Computed the noise terms, now solving the perturbation equations ...')
        delphi_soln = np.zeros(N)  # \delta\hat{\phi}
        ddelphidtau_soln = np.zeros(N)  # d{\delta\hat{\phi}}/d\tau
        delrhor_soln = np.zeros(N)  # \delta\hat{\rho}_r
        Psir_soln = np.zeros(N)  # \hat{\Psi}_r
        varphi_soln = np.zeros(N)  # \hat{\varphi}
        delphi_soln[0], ddelphidtau_soln[0], delrhor_soln[0], Psir_soln[0], varphi_soln[0] = IC
        newIC = IC
        for i in range(N-1):
            bgICs = [phi_bg[i], dotphi_bg[i], ddotphi_bg[i],
                         epsH_bg[i], etaH_bg[i], H_bg[i], T_bg[i],Q_bg[i]] #Initial conditions for the background
            derivs = self.EOMs(newIC, bgICs, tauvals[i], c, m) #Solve the perturbation equations
                # Update the solutions:
            ddelphi = derivs[0] * dtau
            ddelphidtau = derivs[1] * dtau
            ddelrhor = derivs[2] * dtau
            dPsir = derivs[3] * dtau
            dvarphi = derivs[4] * dtau
            delphi_soln[i+1] = delphi_soln[i] + ddelphi
            ddelphidtau_soln[i+1] = ddelphidtau_soln[i] + \
                    ddelphidtau + Noise_th[i] + Noise_qu[i]
            delrhor_soln[i+1] = delrhor_soln[i] + ddelrhor
            Psir_soln[i+1] = Psir_soln[i] + dPsir
            varphi_soln[i+1] = varphi_soln[i] + dvarphi
            newIC = [delphi_soln[i+1], ddelphidtau_soln[i+1],
                         delrhor_soln[i+1], Psir_soln[i+1], varphi_soln[i+1]]
        #Compute the power spectrum:
        hatR2= model.Rpowerspec(
                delphi_soln, delrhor_soln, Psir_soln, varphi_soln, H_bg, T_bg, dotphi_bg)**2
        return hatR2

    def Pool_solver(self, vals, n_cores=0, verbose=False):
        tauvals=self.tauvals
        Q_bg=self.Q_bg
        pool_count=os.cpu_count()
        if n_cores > pool_count: #print error
            print('Error: n_cores is greater than the number of cores available')
        else:
            if n_cores==0:
                n_cores=pool_count-2
            with mp.Pool(processes=n_cores) as pool:
                res = pool.map(self.solve, vals)
                hatR2run = np.array(res)
                hatR2avg = np.mean(hatR2run, axis=0)
                hatR2std = np.std(hatR2run, axis=0)
                index_hor = np.argmin(np.abs(tauvals)) #The index for which \tau=0, i.e. "Ne_inflation" e-folds before the end of inflation
                if verbose:
                    print(index_hor)
                hatR2avg_hor = 1/(2*np.pi**2)*hatR2avg[index_hor]
                hatR2std_hor = 1/(2*np.pi**2)*hatR2std[index_hor]
            return Q_bg[index_hor],hatR2avg_hor, hatR2std_hor



class Scalar_Dissipation_Function:
    def __init__(self, model, Qvals, ph0s, hatR2avg, hatR2std, Nruns, Ne_max, Ne_len, tauvals, Ne_inflation, c, m,Qconstant=True):
        self.Qvals = Qvals
        self.ph0s = ph0s
        self.hatR2avg = hatR2avg
        self.hatR2std = hatR2std
        self.Nruns = Nruns
        self.model = model
        self.c = c
        self.m = m
        self.Ne_max = Ne_max
        self.Ne_len = Ne_len
        self.tauvals = tauvals
        self.Ne_inflation = Ne_inflation
        self.Qconstant = Qconstant
    """Computes the scalar dissipation function of the curvature perturbations for a given c and m. This is simply defined as the ratio betweeen the numerically
    obtained power spectrum divided by the analytic power spectrum valide for c=0, both evaluated "Ne_inflation" e-folds before the end of inflation."""
    def scalar_dissipation_function(self, analytic_power_spectrum=False,deltaR2_analytic =[]):
        Nruns = self.Nruns
        hatR2avg = self.hatR2avg
        hatR2std = self.hatR2std
        model = self.model
        Qvals = self.Qvals
        ph0s = self.ph0s
        Ne_max = self.Ne_max
        Ne_len = self.Ne_len
        tauvals = self.tauvals
        Ne_inflation = self.Ne_inflation
        Qconstant = self.Qconstant
        if analytic_power_spectrum:
            DeltaR2_analytic = deltaR2_analytic
        else:
            DeltaR2_analytic = np.zeros(len(Qvals))
            # Computing the analytic power spectrum:
            for i in trange(len(Qvals)):
                if Qconstant:
                    Bg = Background(model, ph0s[i], Qvals[i])
                else:
                    Bg = Background(model, ph0s[i], Qvals[i], Qconstant=False,Qvariables=[self.c,self.m])
                DeltaR2_analytic[i] = Bg.analytic_power_spectrum(Ne_max, Ne_len, tauvals, Ne_inflation)
            print('Analytic power spectrum computed!')
        # Computing the growth factor, its signal and noise:
        GQ_signal = hatR2avg/DeltaR2_analytic
        GQ_noise = hatR2std/DeltaR2_analytic*1/np.sqrt(Nruns)
        return GQ_signal, GQ_noise

    #Complicated function to fit the growth factor for a positive c:
    def GQ_fit_func_positive_c_log(self, x, a1,a2,a3,a4):
        y=np.log10(1+x)
        return 10**(a1*y+a2*y**2+a3*y**3+a4*y**4)
    # Simple function to fit the growth factor for a positive c:
    def GQ_fit_func_positive_c_pol(self, x, exp1, exp2, a1, a2):
        return 1+a1*x**(exp1)+a2*x**(exp2)

    # Function to fit the growth factor for a negative c:
    def GQ_fit_func_negative_c_pol(self, x, exp1, exp2, exp3, a1, a2):
        return (1+a1*x**exp1)/(1+a2*x**exp2)**exp3

    #Function that determines the best fitting function for the growth factor using the curve_fit function from scipy.optimize:
    def scalar_dissipation_function_fit(self, method='log', make_your_own_bounds=False, lower_bounds=0, upper_bounds=0, output_data=False, analytic_power_spectrum=False, deltaR2_analytic=[]):
        global popt, pcov, low_bounds, up_bounds
        Qvals = self.Qvals
        c = self.c
        GQ_signal, GQ_noise = self.scalar_dissipation_function(analytic_power_spectrum, deltaR2_analytic)
        if c < 0:
            if make_your_own_bounds:
                low_bounds = lower_bounds
                up_bounds = upper_bounds
            else:
                low_bounds = [0.1, 0.1, 0.1,0, 0 ]
                up_bounds = [10, 10, 10, 10, 10]
            popt, pcov = curve_fit(self.GQ_fit_func_negative_c_pol, Qvals,
                                  GQ_signal, sigma=GQ_noise, bounds=(low_bounds, up_bounds))
        if method == 'log' and c > 0:
            if make_your_own_bounds:
                low_bounds = lower_bounds
                up_bounds = upper_bounds
            else:
                low_bounds = [ -15, -15, -15,-15]
                up_bounds = [ 10, 10, 10,10]
            popt, pcov = curve_fit(self.GQ_fit_func_positive_c_log, Qvals,
                                   GQ_signal, sigma=GQ_noise, bounds=(low_bounds, up_bounds))

        if method == 'pol' and c > 0:
            if make_your_own_bounds:
                low_bounds = lower_bounds
                up_bounds = upper_bounds
            else:
                low_bounds = [0.01, 0.01, 0, 0]
                up_bounds = [10, 10, 10, 10]
            popt, pcov = curve_fit(self.GQ_fit_func_positive_c_pol, Qvals,
                                   GQ_signal, sigma=GQ_noise, bounds=(low_bounds, up_bounds))
        perr = np.sqrt(np.diag(pcov))
        if output_data:
            return popt, perr, GQ_signal, GQ_noise
        else:
            return popt, perr
