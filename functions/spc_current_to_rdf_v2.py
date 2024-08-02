import numpy as np
import xarray as xr
import astropy.units as u
import astropy.constants as c

def _E(mv_lo, mv_hi):
    '''
    This function calculated the complete elliptinc integral of the second kind.
    This function is a supplemental function. More details about this intergral 
    can be found in the SWEAP Data Users guide: http://sweap.cfa.harvard.edu/sweap_data_user_guide.pdf
    and the supplement of this paper: https://ui.adsabs.harvard.edu/abs/2010ApJ...722L..75S/abstract 

    Parameters:
    -----------
    mv_lo : np.ndarray
        Low modulator voltage
    mv_hi : np.ndarray
        High modulator voltage

    Returns:
    --------
    E : np.ndarray
        The complete elliptic integral of the second kind using the Hastings
        Algorithm
    '''
    k = np.sqrt((mv_hi - mv_lo) / (mv_hi))     # This is the k from the SPC paper.
    x = 1.0 - k**2
    
    # Coefficients taken from SWEAP Report
    a1=0.44325141463
    a2=0.06260601220
    a3=0.04757383546
    a4=0.01736506451
    b1=0.24998368310
    b2=0.09200180037
    b3=0.04069697526
    b4=0.00526449639

    # Complete elliptic integral
    E1 = 1 + x*( a1 + x*(a2 + x*(a3 + x*a4)))
    E2 = x*(b1 + x*(b2 + x*(b3 + x*b4)))*np.log(1./x)

    E = E1 + E2
    return(E)


def derive_reduced_vdf(spc_ds, DATASET=True):
    '''
    Derive the reduced VDF from the SPC Data.

    Parameters:
    -----------
    spc_ds : xr.Dataset
        Xarray.Dataset containing the ion VDF information.
    
    Returns:
    --------
    ds : xr.Dataset
        spc_ds appended with the derived quantities.
    '''
    # Copy the initial xarray.Dataset
    ds = spc_ds.copy()

    total_current = ds.a_current + ds.b_current + ds.c_current + ds.d_current

    # Flow angles from SPC
    # NOTE: After looking at the flow angle calculations form Mike, there is a factor of two that I am uncertain where
    # it comes from. Following Tony's SPC paper the flow angles are lambda * (plate differences)/total, where the plate differences
    # are the left-right and up-down current differences, and lambda is given to be ~1.009 = pi/2 * (r_la/d_la), where r is radius and d is distance 
    # of the limiting appature. Therefore, following Tony's calculation to convert to degrees would be 180/pi. But 
    # to keep with Mike's convention we need a 1/2 included.   
    #
    # I have preformed a recent test to compare the flow angle derived in the L2 SPC data set and the derived flow angle from the currents.
    # It is noted that the factor of 2 is needed to ensure agreement between the two data sets. Therefore, for now I will include the 
    # factor of 2! I am uncertain as to why this is the case.
    lamda = 1.009

    # The flow angles for each voltage bin.
    azimuth   = (180/(2*np.pi)) * lamda * (((ds.a_current + ds.d_current) - (ds.b_current + ds.c_current))/total_current)   # In Degrees!
    elevation = (180/(2*np.pi)) * lamda * (((ds.c_current + ds.d_current) - (ds.a_current + ds.b_current))/total_current)   # In Degrees

    mv_mid = 0.5 * (ds.mv_hi.data + ds.mv_lo.data)      # This is obsolete...

    # Flow angle. 
    #
    # NOTE: We need to make sure any spurious peaks are removed if not the max value will be incorrect.
    # This should be checked in future versions of this code. 
    peak_azimuth   = (180/(2*np.pi))* lamda *(((np.nanmax(ds.a_current.data, axis=1) + np.nanmax(ds.d_current.data, axis=1)) - (np.nanmax(ds.b_current.data, axis=1) + np.nanmax(ds.c_current.data, axis=1)))/np.nanmax(total_current.data, axis=1))   # In Degrees!
    peak_elevation = (180/(2*np.pi))* lamda *(((np.nanmax(ds.c_current.data, axis=1) + np.nanmax(ds.d_current.data, axis=1)) - (np.nanmax(ds.a_current.data, axis=1) + np.nanmax(ds.b_current.data, axis=1)))/np.nanmax(total_current.data, axis=1))   # In Degrees

    # Effective width of energy interval
    eff_width      = (ds.mv_hi.data - ds.mv_lo.data)/2.
    center_voltage = (ds.mv_hi.data + ds.mv_lo.data)/2.
    
    vel_lo  = np.sqrt(2 * c.e.si * ds.mv_lo.data * (u.Volt) / c.m_p).to(u.km/u.s)
    vel_hi  = np.sqrt(2 * c.e.si * ds.mv_hi.data * (u.Volt) / c.m_p).to(u.km/u.s)
    
    # Using the complete elliptic intergral
    vel_mid = np.sqrt(2 * c.e.si * ds.mv_hi.data * (u.Volt) / c.m_p).to(u.km/u.s) * (2.0/np.pi) * _E(ds.mv_lo.data, ds.mv_hi.data)

    # Calculate the differential velocity
    dv = np.sqrt((4 * (c.e.si)/(c.m_p) * (center_voltage) * (u.Volt)).to(u.km**2/u.s**2) - vel_mid**2)

    # Reduced rdf value, this is not quite accurate as it does not take into account the geometric factor. 
    # However, the geometric factor is removed once the re-normalization in the fitting is acheived. 
    rdf = total_current.data / (c.e.si * vel_mid * dv)

    # get the x and y components
    vel_z_tan_az = vel_mid * np.tan(np.radians(azimuth.data))
    vel_z_tan_el = vel_mid * np.tan(np.radians(elevation.data))     # There was a negative sign here that should not be needed! KP defined elevation flow angle
                                                                    # as (a + b) - (c + d)/tot. This needs a sign to adhere with convention
                                                                    # In this case elevation is (c + d) - (a + b)/tot. HENCE NO SIGN CHANGE NEEDED!

    if DATASET:
        names = ['azimuth', 'elevation', 
                 'total_current', 'mv_mid', 
                 'vel_lo', 'vel_hi', 'vel_mid',
                 'vel_x', 'vel_y', 'rdf']
        
        array_values = [azimuth.data,
                        elevation.data,
                        total_current.data,
                        mv_mid,
                        vel_lo.value,
                        vel_hi.value,
                        vel_mid.value,
                        vel_z_tan_az.value,
                        vel_z_tan_el.value,
                        rdf.value]

        array_units = ['Radians',
                       'Radians',
                       'Volts',
                       'Volts',
                       vel_lo.unit,
                       vel_hi.unit,
                       vel_mid.unit,
                       vel_z_tan_az.unit,
                       vel_z_tan_el.unit,
                       rdf.unit]

        for i, name in enumerate(names):
            DA = xr.DataArray(array_values[i],
                              dims=['time', 'dim'],
                              coords=dict(time = ds.time.data,
                                          dim = np.arange(32)),
                                          #mv_lo=(['time', 'vdim'], ds.mv_lo.data)),
                              attrs=dict(UNIT = array_units[i])
                             )
            ds[name] = DA

        ds['peak_azimuth'] = (['time'], peak_azimuth)
        ds['peak_elevation'] = (['time'], peak_elevation)

        return(ds)
    else:
        return(0)