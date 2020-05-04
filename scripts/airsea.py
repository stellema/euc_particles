# -*- coding: utf-8 -*-
"""
created: Wed Apr 15 11:39:07 2020

author: Annette Stellema (astellemas@gmail.com)

Bulk flux code modified from:
https://github.com/pyoceans/python-airsea/blob/master/airsea/atmosphere.py

NCI CMS ERA-Interim details:
http://climate-cms.wikis.unsw.edu.au/ERA_INTERIM

NCI CMS ERA-Interim variables:
https://docs.google.com/spreadsheets/d/1qnQC_Ki5IAwZPD9viV79tfenemPGoYWKfDa5vEwDl90/pubhtml#

ERA-Interim Specific humidity at surface conversion advice:
https://confluence.ecmwf.int/display/CKB/ERA-Interim%3A+documentation
    and
https://www.ecmwf.int/en/elibrary/9221-part-iv-physical-processes

Vapour pressure constants:
https://www.eas.ualberta.ca/jdwilson/EAS372_13/Vomel_CIRES_satvpformulae.html

Specific humidity and vapour pressure formulas (specific to dew point temps):
https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html


ds = xr.open_dataset(cfg.data/'{}_uas_climo.nc'.format(data[0])).uas
"""
import cfg
import tools
import numpy as np
import xarray as xr
import pandas as pd
from warnings import warn
from scipy.interpolate import griddata


def reduce(ds, mean_t=False, res=0.1, interp='', x2=[], y2=[]):
    """Slice, rename and/or time average datasets."""
    if hasattr(ds, 'xu_ocean'):
        ds = ds.rename({'Time': 'time', 'yu_ocean': 'lat',
                        'xu_ocean': 'lon'})
        ds = ds.sel(st_ocean=2.5)
    elif hasattr(ds, 'xt_ocean'):
        ds = ds.rename({'Time': 'time', 'yt_ocean': 'lat',
                        'xt_ocean': 'lon'})
        ds = ds.sel(st_ocean=2.5)

    if mean_t and hasattr(ds, 'time'):
        ds = ds.mean('time')

    if interp != '':
        if x2 == []:
            y2 = np.arange(-15, 14.9 + res, res)
            x2 = np.arange(120, 294.9 + res, res)

        x1 = ds.lon.values
        y1 = ds.lat.values
        X1, Y1 = np.meshgrid(x1, y1)
        X2, Y2 = np.meshgrid(x2, y2)
        if ds.ndim == 3:
            tmp = np.empty((len(ds.time), len(y2), len(x2)))*np.nan
            for t in range(len(ds.time)):
                tmp[t] = griddata((X1.flatten(), Y1.flatten()),
                                  ds[t].values.flatten(), (X2, Y2),
                                  method=interp)
        else:

            tmp = griddata((X1.flatten(), Y1.flatten()),
                           ds.values.flatten(), (X2, Y2),
                           method=interp)

        coords = [y2, x2] if ds.ndim == 2 else [ds.time, y2, x2]
        dims = ['lat', 'lon'] if ds.ndim == 2 else ['time', 'lat', 'lon']
        ds = xr.DataArray(tmp, coords=coords, dims=dims)

    return ds


def vapour_pressure(Td):
    """Water vapour pressure using Teten's formula.

    Multiplies by 100 to convert millibar to Pa.

    Args:
        T (array like): Dew point temperature.

    Returns:
        Water vapour pressure (array like).

    """
    if (Td <= 250).all():
        Td = Td + 273.16
    a1 = 6.1078
    a3 = 17.2693882
    a4 = 35.86
    T0 = 273.16

    return (a1 * np.exp(a3 * ((Td - T0)/(Td - a4))))*100


def specific_humidity_from_p(Td, p):
    """Speciﬁc humidity from water vapour pressure.

    Args:
        T (array like): Dew point temperature.
        p (array like): Surface pressure [Pa].

    Returns:
        Specific humidity [kg/kg] (array like).

    """
    R_dry = 287  # Gas constant for dry air [J K^-1 kg^-1].
    R_vap = 461  # Gas constant for water vapour [J K^-1 kg^-1].
    omega = R_dry/R_vap  # Mixing ratio [kg/kg].

    # Convert to Kelvins.
    if (Td <= 250).all():
        Td = Td + 273.16

    return ((omega * vapour_pressure(Td)) /
            (p - ((1 - omega) * vapour_pressure(Td))))


def flux_data(dat='jra55', res=0.1, mean_t=True, interp=''):

    # Observed vector wind at height z_U, in m/s.
    u_O = xr.open_dataset(cfg.data/'{}_uas_climo.nc'.format(dat))
    v_O = xr.open_dataset(cfg.data/'{}_vas_climo.nc'.format(dat))
    U_O = (reduce(u_O.uas, mean_t, res, interp) +
           1j * reduce(v_O.vas, mean_t, res, interp))

    # Observed temperature at height z_T, in degrees Celsius
    T_O = xr.open_dataset(cfg.data/'{}_tas_climo.nc'.format(dat))
    T_O = reduce(T_O.tas, mean_t, res, interp)

    # Observed specific humidity at height z_q, in kg/kg.
    if dat == 'jra55':
        q_ = xr.open_dataset(cfg.data/'{}_huss_climo.nc'.format(dat))
        q_O = reduce(q_.huss, mean_t, res, interp)
    else:
        ps = xr.open_dataset(cfg.data/'{}_ps_climo.nc'.format(dat))
        ps = reduce(ps.ps, mean_t, res, interp)
        td = xr.open_dataset(cfg.data/'{}_ta2d_climo.nc'.format(dat))
        td = reduce(td.ta2d, mean_t, res, interp)
        q_O = specific_humidity_from_p(td, ps)

    # Sea level pressure, in hectopascal (hPa). [Original units Pa]
    SLP = xr.open_dataset(cfg.data/'{}_psl_climo.nc'.format(dat))
    SLP = reduce(SLP.psl, mean_t, res, interp)

    # Convert from Kelvin to Celsius.
    if (T_O >= 250).all():
        T_O = T_O - 273.16

    # Convert from Pa to hPa.
    if (SLP > 1e5).all():
        SLP = SLP*0.01

    # OFAM3 SST
    SST = xr.open_dataset(cfg.ofam/'ocean_temp_1981-2012_climo.nc')

    # Surface vector current, in m/s.
    SSU_u = xr.open_dataset(cfg.ofam/'ocean_u_1981-2012_climo.nc')
    SSU_v = xr.open_dataset(cfg.ofam/'ocean_v_1981-2012_climo.nc')

    # if interp != '':
    SST = reduce(SST.temp, mean_t, res, interp='')
    SSU = (reduce(SSU_u.u, mean_t, res, interp='') +
           1j * reduce(SSU_v.v, mean_t, res, interp=''))
    # else:

    #     lats = np.append(np.arange(-15, 15, 1.25), 14.9)
    #     lons = np.arange(120, 296, 1.25)
    #     SSU_u = reduce(SSU_u.u, mean_t=mean_t, res=res, interp='cubic', x2=lons, y2=lats)
    #     SSU_v = reduce(SSU_v.v, mean_t=mean_t, res=res, interp='cubic', x2=lons, y2=lats)
    #     # SSU_u = SSU_u.interp(lat=lats, lon=lons)
    #     # SSU_v = SSU_v.interp(lat=lats, lon=lons)
    #     SSU = SSU_u + 1j * SSU_v
    #     lats = np.append(np.arange(-14.95, 15, 1.25), 14.95)
    #     lons = np.append(np.arange(120.1, 295, 1.25), 295)
    #     SST = reduce(SST.temp, mean_t=mean_t, res=res, interp='cubic', x2=lons, y2=lats)
    #     # SST = reduce(SST.temp, mean_t=mean_t, interp='')
    #     # SST = SST.interp(lat=lats, lon=lons)

    #     SST = SST.assign_coords({'lat': SLP.lat.values, 'lon': SLP.lon.values})
        # y2 = np.append(SLP.lat[:-1].values, 15)
        # x2 = np.append(SLP.lon[:-1].values, 294.9)
        # SSU_u = reduce(SSU_u.u, mean_t=mean_t, interp='cubic',
        #                 y2=y2, x2=x2)
        # SSU_v = reduce(SSU_v.v, mean_t=mean_t, interp='cubic',
        #                 y2=y2, x2=x2)
        # SSU = SSU_u + 1j * SSU_v
        # y2 = np.append(SLP.lat[:-1].values, 14.95)
        # x2 = np.append(SLP.lon[:-1].values, 295)
        # x2[0]=120.1
        # y2[0]=-14.95
        # SST = reduce(SST.temp, mean_t=mean_t, interp='cubic',
        #               y2=y2, x2=x2)

    dl = (U_O, T_O, q_O, SLP, SST, SSU)

    if not all(x.shape == dl[0].shape for x in dl):
        warn('Array shapes unequal (see print out).')
        for d in dl:
            print('lat={:.2f}-{:.2f}, lon={:.2f}-{:.2f} {}'
                  .format(d.lat[0].item(), d.lat[-1].item(),
                          d.lon[0].item(), d.lon[-1].item(), d.shape))
    return dl


def bulk_fluxes(U_O, T_O, q_O, SLP, SST, SSU, z_U=10, z_T=2,
                z_q=2, N=5, result='TAU'):
    """Air-sea bulk turbulent fluxes computation.

    Definitions
    -----------
    Tau : Momentum, wind stress, in N \cdot m^{-2}, or
          kg \cdot m \cdot s^{-2} \cdot m^{-2}
    Q_H : Sensible heat, in W \cdot m^{-2}, or
          J \cdot s^{-1} \cdot m^{-2}
    Q_E : Latent heat, in W \cdot m^{-2}, or
          J \cdot s^{-1} \cdot m^{-2}
    E   : Evaporation, in mg \cdot s^{-1} \cdot m^{-2}

    Given
    -----
    U_O : Observed vector wind at height z_U
    T_O : Observed temperature at height z_T
    q_O : Observed specific humidity at height z_q
    SLP : Sea level pressure
    SST : Sea surface temperature
    SSU : Surface vector current

    Parameters
    ----------
    U_O : array like
        Observed vector wind at height z_U, in m/s. Accepts
        velocity given in complex notation.
    T_O : array like
        Observed temperature at height z_T, in degrees Celsius
    q_O : array like
        Observed specific humidity at height z_q, in kg/kg.
    SLP : array like
        Sea level pressure, in hectopascal (hPa).
    SST : array like
        Sea surface temperature, in degrees Celsius.
    SSU : array like
        Surface vector current, in m/s. Accepts velocity given in
        complex notation.
    z_U, z_T, z_q : float, array like, optional
        Height of the observed vector wind, temperature and
        specific humidity, in m. Default value is 10 m.
    N : integer, optional
        Number of iterations to calculate bulk parameters.
    result : string, optional
        Determines if either only 'fluxes' (default) or if 'all'
        bulk calculations are returned.
    units : dictionary, optional
        Sets the units for the variables.

    Returns
    -------
    Tau, Q_H, Q_E : array like
        Momentum, sensible and latent heat fluxes.
    dU10, dtheta10, dq10, L, zeta: array like, optional :
        If `result` is set to 'all', additionally to the fluxes, ...

    References
    ----------
    Large, W. G. and S. Yeager (2004). Diurnal to decadal global forcing
    for ocean and sea-ice models: The data sets and flux climatologies.
    Technical note NCAR/TN-460+STR, NCAR.
    Large, W. G. (2006). Surface fluxes for practioners of global ocean
    data assimilattion. In E. Chassignet and J. Verron (Eds.), Ocean
    weather and forecasting, pp. 229–270. Heidelberg: Springer.
    """
    ###########################################################################
    # CONSTANTS AND PARAMETERS
    ###########################################################################
    GAMMA = 0.01    # Adiabaic lapse rate for dry air.
    R_gas = 287.04  # Dry air gas constant [J/kg/K].
    KAPPA = 0.4     # von Karman constant.
    LAMBDA = 2.5e6  # Latent heat of vaporization [J/kg].
    g = 9.816       # Gravitational acceleration [m/s**2].
    cp = 1003.5     # Specific heat capacity of dry air [J/kg/K].

    ###########################################################################
    # FUNCTIONS
    ###########################################################################

    def pot_temperature(T, z):
        """Potential temperature.

        Parameters
        ----------
        T : array like
            Temperature.
        z : array like
            Height of temperature.
        Returns
        -------
        theta : array like
            Potential temperature, in degrees Celsius
        """
        return T + GAMMA * z

    def virtual_pot_temperature(theta, q):
        """Virtual potential air temperature.

        Parameters
        ----------
        theta : array like
            Potential temperature.
        q : array like
            Specific humidity.
        Returns
        -------
        theta_V : array like
            Virtual air temperature
        """
        return theta * (1 + 0.608 * q)

    def density_air(SLP, theta_V, SLP_unit='hPa', T_unit='degC'):
        """Density of air.

        Parameters
        ----------
        SLP : array like
            Sea level pressure.
        theta_V : array like
            Virtual temperature.
        SLP_unit : string, optional
            Sets wether sea level pressure is given in bar ('bar'),
            hectopascal ('hPa', default), millibar ('mbar'),
            kilopascal ('kPa') or standard atmosphere ('atm').
        T_unit : string, optional
            Sets wether sea surface temperature is given in degrees
            Celsius ('degC', default) or Kelvin ('K').
        Returns
        -------
        rho_air : array like :
            Density of air.
        """
        if SLP_unit == 'bar':
            # Converts SLP from bar to Pa
            SLP = SLP * 1e5
        elif SLP_unit in ['hPa', 'mbar']:
            # Converts SLP from hPa to Pa
            SLP = SLP * 1e2
        elif SLP_unit == 'kPa':
            # Converts SLP from kPa to Pa
            SLP = SLP * 1e3
        elif SLP_unit == 'atm':
            # Converts SLP from atm to Pa
            SLP = SLP * 1.01325e5
        elif SLP_unit == 'Pa':
            SLP = SLP * 1.

        if T_unit == 'degC':
            # Converts temperature from degrees Celsius to Kelvin.
            theta_V = theta_V + 273.16

        return SLP / (R_gas * theta_V)

    def humidity_sat(SST, rho, q1=0.98, q2=640380, q3=-5107.4,
                     SST_unit='degC'):
        """Parameterized saturated humidity.

        Parameters
        ----------
        SST : array like
            Sea surface temparature.
        rho : array like
            Air density, in kg*m**(-3).
        q1, q2 : float, optional
            Specicic coefficients for sea-water.
        SST_unit : string, optional
            Sets wether sea surface temperature is given in degrees
            Celsius ('degC', default) or Kelvin ('K').
        Returns
        -------
        q_sat : array like
            Saturated humidity over seawater, in kg/kg
        """
        if SST_unit == 'degC':
            # Converts SST from degrees Celsius to Kelvin.
            SST = SST + 273.16

        return q1 * q2 / rho * np.exp(q3 / SST)

    def u_star(dU, CD):
        """Calculate friction velocity u*.

        Paramaters
        ----------
        dU : array like
            Difference between wind velocity and sea surface velocity,
            in m/s.
        CD : array like
            Drag coefficient.
        Returns
        -------
        u_star : array like
            Friction velocity.
        """
        return (CD ** 0.5) * dU

    def t_star(dtheta, CD, CH):
        """Calculate turbulent fluctuations of potential temperature.

        Parameters
        ----------
        dtheta : array like
            Difference between potential temperature and sea surface
            temperature, in degrees Celsius.
        CD : array like
            Drag coefficient.
        CH : array like
            Stanton number.
        Returns
        -------
        theta_star : array like
            Turbulent fluctuations of potential temperature.
        """
        return CH / (CD ** 0.5) * dtheta

    def q_star(dq, CD, CE):
        """Calculate turbulent fluctuations of specific humidity.

        Parameters
        ----------
        dq : array like
            Difference between specific humidity and saturated humidity
            at sea surface, in mg/kg.
        CD : array like
            Drag coefficient.
        CE : array like
            Dalton number.
        Returns
        -------
        q_star : array like
            Turbulent fluctuations of specific humidity.
        """
        return CE / (CD ** 0.5) * dq

    def stability_atmosphere(z, L):
        """Atmospheric stability.

        Parameters
        ----------
        z : array like
            Height, in m.
        L : array like
            Monin-Obhukhov length
        Returns
        -------
        zeta : array like
            Atmospheric stability.
        """
        return z / L

    def Monin_Obukhov_Length(u_star, theta_star, q_star, theta_V, q):
        """Calculate the Monin-Obukhov length.

        The length is used to describe the effects of buoyancy on turbulent
        flows, particularly in the lower tenth of the atmospheric boundary
        layer.
        Parameters
        ----------
        u_star : array like
            Frinction velocity u*, in m/s.
        theta_star : array like
            Scaling temperature, in degrees Celsius.
        q_star : array like
            Scaling specific humidity, in mg/kg.
        theta_V : array like
            Virtual potential temperature, in degrees Celsius.
        q : array like
            Specific humidity, in mg/kg.
        Returns
        -------
        L : array like
            Monin-Obukhov length
        References
        ----------
        Large, W. G. (2006). Surface fluxes for practioners of global
        ocean data assimilattion. In E. Chassignet and J. Verron (Eds.),
        Ocean weather and forecasting, pp. 229–270. Heidelberg:
        Springer.
        http://en.wikipedia.org/wiki/Monin%E2%80%93Obukhov_length
        """
        # Converts virtual potential temperature from degres Celsius to Kelvin
        theta_V = theta_V + 273.16
        B0 = g * (theta_star / theta_V + q_star / (q + 0.608**(-1)))
        return u_star**2 / (KAPPA * B0)

    def Psi(zeta, result='both'):
        """Empirical stability functions.

        The returned values of Psi_M and Psi_S are used to bring
        observational measurements of wind speed, potential temperature and
        humidity from non-netural profiles to neutral profiles. They are the
        integrals of the dimensionless flux profiles of momentum,
        \Psi_M(\zeta), and of the scalars heat and moisture, \Psi_S(\zeta).
        Parameters
        ----------
        zeta : array like
            Atmospheric stability.
        result : string
            Determines if either the 'momentum', 'scalar' or 'both' (default)
            dimensionless flux profiles are returned.
        Returns
        -------
        Psi_M, Psi_S : array like
            According to the 'result' parameters, function returns
            either \Psi_M(\zeta), and/or \Psi_S(\zeta).
        Reference
        ---------
        Paulson, C. A., 1970: The Mathematical Representation of Wind
        Speed and Temperature Profiles in the Unstable Atmospheric
        Surface Layer. J. Appl. Meteor., 9, 857–861.
        """
        mask_stable = (zeta >= 0)
        mask_unstable = (zeta < 0)
        # Initializes variables
        Psi_M = zeta * 0
        Psi_S = zeta * 0
        # Calculates Psi_M and Psi_S for the stable case (zeta > 0)
        Psi_M += -5 * zeta * mask_stable
        Psi_S += -5 * zeta * mask_stable
        # Calculates Psi_M and Psi_S for the unstable case (zeta < 0). It is
        # important to note that X = (1 - 16 * zeta)**0.25, but since zeta < 0,
        # we use absolute values and then mask them out to ensure proper
        # calculations.
        X = (1 + 16 * abs(zeta)) ** 0.25
        Psi_M += 2 * np.log((1 + X**2) / 2) * mask_unstable
        Psi_S += (np.log((1 + X**2) / 2) + np.log((1 + X) / 2) - 2
                  * np.arctan(X) + 0.5 * np.pi) * mask_unstable
        if result == 'both':
            return Psi_M, Psi_S
        elif result == 'momentum':
            return Psi_M
        elif result == 'scalar':
            return Psi_S
        else:
            raise ValueError('Invalid result type `{}`.'.format(result))

    def Q_E(E):
        """Convert evaporation to latent heat of evaporation.

        Parameters
        ----------
        E : array like
            Evaporation, in mg \cdot s^{-1} \cdot m^{-2}.
        Returns
        -------
        Q_E : array like
            Latent heat of evaporation, in W \cdot m^{-2}.
        """
        return LAMBDA * E

    def law_of_the_wall(u_star, z, z0, Psi=0):
        """Law of the wall for wind, temperature or humidity profiles.

            DU(z) = (u_star / KAPPA) * ln(z / z0 - Psi)
        Parameters
        ----------
        u_star : float
            Depending on the application, this parameter may be the
            friction velocity (in m/s), scaling temperature (in K or
            degC), scaling specific humidity (in mg/kg).
        z : array like
            Height above sea level (in m).
        z0 : float
            Roughness length (in m).
        Psi : float
            Empirical correction for stability. Neutral profiles have
            Psi=0.
        Returns
        -------
        DU : array like
            Either wind velocity, temperature or specific humidity at
            height z.
        """
        return u_star / KAPPA * (np.log(z / z0) - Psi)

    def C_D(dU):
        """Calculate the drag coefficient using multiple regression parameters.

        Parameters
        ----------
        dU : array like
            Difference between wind and sea surface velocities,
            in m/s.
        Returns
        -------
        CD : array like
            Drag coefficient.
        """
        # CD = u_star**2 / dU ** 2
        a1 = 0.00270
        a2 = 0.000142
        a3 = 0.0000764
        return a1 / dU + a2 + a3 * dU

    def C_E(dU):
        """Calculate the Dalton number.

        Parameters
        ----------
        dU : array like
            Difference between wind and sea surface velocities, in m/s.
        Returns
        -------
        CE : array like
            Dalton number.
        """
        return 0.0346 * C_D(dU) ** 0.5

    def C_H(dU, zeta=0):
        """Calculate the Stanton number.

        Parameters:
        dU : array like
            Difference between wind and sea surface velocities, in m/s.
        zeta : array like
            Atmospheric stability.
        Returns
        -------
        CH : array like
            Stanton number
        """
        return (0.0180 * C_D(dU) ** 0.5 * (zeta >= 0) +
                0.0327 * C_D(dU) ** 0.5 * (zeta < 0))

    if isinstance(U_O, xr.DataArray):
        dims, coords = U_O.dims, U_O.coords
        U_O, T_O, q_O = U_O.values, T_O.values, q_O.values
        SLP, SST, SSU = SLP.values, SST.values, SSU.values

    # STEP ZERO: Checks for proper units.
    default_units = dict(U_O='m s-1', T_O='degC', q_O='kg kg-1', SLP='hPa',
                         SST='degC', SSU='m s-1', z_U='m', z_T='m', z_q='m')

    # FIRST STEP: Assume theta(z_u) = theta(z_theta) and q(z_u) = q(z_q),
    # compute potential temperature and virtual potential temperature, sea
    # surface humidity, difference between wind speed and surface current
    # speed, difference between potential temperature and sea surface
    # temperature, observed specific humidity and sea surface humidity.
    theta_O = pot_temperature(T_O, z_T)
    theta_O_v = virtual_pot_temperature(theta_O, q_O)
    SSq = humidity_sat(SST, density_air(SLP, theta_O_v))
    DU = U_O - SSU
    dU = abs(DU)
    dtheta = theta_O - SST
    dq = q_O - SSq

    # SECOND STEP: Assume neutral stability (\zeta = 0) and observations at
    # z=10m. Calculate the transfer coefficients and initial turbulent scales.
    zeta = 0
    dU10, dtheta10, dq10 = dU, dtheta, dq
    for i in range(N):
        # Some checks first:
        try:
            dU10 = np.where(dU10 < 1, 1, dU10)
        except:
            pass

        CD = C_D(dU10)
        CH = C_H(dU10, zeta=zeta)
        CE = C_E(dU10)

        US = u_star(dU10, CD)
        thetaS = t_star(dtheta10, CD, CH)
        qS = q_star(dq10, CD, CE)

        q = SSq + dq10
        theta_v = SST + dtheta10 + 0.608 * q
        L = Monin_Obukhov_Length(US, thetaS, qS, theta_v, q)
        zeta = stability_atmosphere(10., L)

        dU10 = dU - law_of_the_wall(US, z_U, 10.,
                                    Psi=Psi(zeta*z_U/10., result='momentum'))
        dtheta10 = dtheta - law_of_the_wall(thetaS, z_T, 10.,
                                            Psi=Psi(zeta*z_T/10.,
                                                    result='scalar'))
        dq10 = dq - law_of_the_wall(qS, z_q, 10.,
                                    Psi=Psi(zeta*z_q/10., result='momentum'))

    # THIRD STEP: Compute the bulk turbulent fluxes.
    rho = density_air(SLP, theta_v)
    tau = rho * US ** 2
    Tau = tau * DU / dU
    QH = rho * cp * US * thetaS
    E = rho * US * qS
    E[E > 0] = np.nan
    QE = Q_E(E)

    TAU = xr.DataArray(Tau, dims=dims, coords=coords)
    if result == 'fluxes':
        return TAU, QH, QE
    elif result == 'TAU':
        return TAU
    elif result == 'all':
        return TAU, QH, QE, dU10, dtheta10, dq10, L, zeta


def prescribed_momentum(u, v, method='static'):
    """Compute wind stress from wind field data Based on Gill (1982).

    Formula and a non-linear drag coefficent based on Large and Pond (1981),
    modified for low wind speeds (Trenberth et al., 1990).
    https://
    au.mathworks.com/matlabcentral/fileexchange/53391-wind-stresses-computation

    Args:
        u (DataArray): Zonal wind at 10 m.
        v (DataArray): Meridional wind at 10 m.

    Returns:
        tx (DataArray): Zonal component of wind stress.
        ty (DataArray): Meridional component of wind stress.

    """
    import math
    p = 1.225  # Air density [kg/m^3].
    # Computation of Wind Stresses.
    if len(u.shape) == 3:
        [nlats, nlons] = u[0].shape
    else:
        [nlats, nlons] = u.shape
    tx = u.copy()*np.nan
    ty = v.copy()*np.nan
    U = np.sqrt(u**2 + v**2)  # Wind speed.
    if method == 'static':
        cd = 0.0013
        tx = cd*p*U*u  # kg/m^3*m/s*m/s = N/m^2.
        ty = cd*p*U*v
    else:
        # cd = U.copy()*np.nan
        if method == 'LARGE_approx':
            drag = pd.read_csv(cfg.data/'cdrag.csv')
        for j in range(nlats):
            for i in range(nlons):
                U = math.sqrt(u[j, i]**2 + v[j, i]**2)
                if ~np.isnan(U):
                    if method == 'GILL':
                        # Random method.
                        if (U <= 1):
                            cd = 0.00218
                        elif (U > 1 or U <= 3):
                            cd = (0.62 + 1.56/U)*0.001
                        elif (U > 3 or U < 10):
                            cd = 0.00114
                        else:
                            cd = (0.49 + 0.065*U)*0.001

                    # YEAGER LARGE (approx).
                    elif method == 'LARGE_approx':
                        drag = pd.read_csv(cfg.data/'cdrag.csv')
                        xi = tools.idx(drag['u'].values, U.item())

                        if (drag['u'][xi] != U[j, i] and
                            xi != 0 and xi != len(drag['u']-1)):

                            x1i = xi if drag['u'][xi] <= U[j, i] else xi - 1
                            x2i = xi + 1 if drag['u'][xi] <= U[j, i] else xi
                            x1, x2 = drag['u'][x1i], drag['u'][x2i]
                            y1, y2 = drag['cd'][x1i], drag['cd'][x2i]
                            b = (x1*y2 - x2*y1)/(x1-x2)
                            m = (y1-y2)/(x1-x2)
                            cd[j, i] = (m*U[j, i] + b)*0.001
                        else:
                            cd = drag['cd'][xi]*0.001

            # Equation.
            tx[j, i] = cd*p*U*u[j, i]  # kg/m^3*m/s*m/s = N/m^2.
            ty[j, i] = cd*p*U*v[j, i]

    return tx, ty
