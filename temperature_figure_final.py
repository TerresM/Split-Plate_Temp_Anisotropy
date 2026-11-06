import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cdflib.xarray
import xarray as xr
import astropy.units as u
import astropy.constants as c

import cmcrameri.cm as cmc

import pickle

from datetime import datetime
from functions.span_flow_angle_calc import flow_angles_span_inst

def Partial_T_anisotropy(Txx, Tzz, b_vec):
    magb = np.linalg.norm(b_vec, axis=1)
    theta = np.arccos(b_vec[:,2]/magb)

    T_perp = Tzz + (Txx - Tzz)/(1 - ((b_vec[:,0]/magb)**2) * np.tan(theta)**2)
    T_para = T_perp + (Txx - Tzz)/(((b_vec[:,0]/magb)**2) * np.sin(theta)**2 - np.cos(theta)**2)

    T_fa = np.vstack([T_perp, T_para])
    return(T_fa)

if __name__ == "__main__":
    file_path = 'selected_dates_updated_v2.pkl'
    with open(file_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    file_path2 = 'updated_flags_intervals_v2.pkl'
    with open(file_path2, 'rb') as f:
        updated_flags2 = pickle.load(f)

    # Flatten the list and apply the date cutoff
    loaded_list = [date for date_list in loaded_dict.values() for date in date_list]
    cutoff = datetime(2023, 1, 1) 
    loaded_list = [d for d in loaded_list if datetime.strptime(d, "%Y-%m-%d") > cutoff]

    T_ani_spc_dict = []
    T_ani_span_dict = []
    T_ani_spc_bimax_dict = []
    T_ani_span_bimax_dict = []
    T_ani_bimax_dict = []
    times_dict = []
    valid_dates = []
    for date in loaded_list:
        year, month, day = date.split('-') 
        print(f"Processing date: {date}")

        # Load Data
        # spc   = cdflib.xarray.cdf_to_xarray(f'./anisotropy_files/Bayesian_MCMC_5D_NCP_DEP/SPC_Anisotropy_{date}_peak_track_T_fit4s_mf37_gflag.cdf', fillval_to_nan=True)
        spc   = cdflib.xarray.cdf_to_xarray(f'./anisotropy_files/Bayesian_MCMC_5D_NCP/SPC_Anisotropy_{date}_peak_track_T_fit4s_mf38_gflag_v2.cdf', fillval_to_nan=True)
        try:
            span  = cdflib.xarray.cdf_to_xarray(f'./psp_data/sweap/spi/L3/spi_sf00/{year}/{month}/psp_swp_spi_sf00_L3_mom_{year}{month}{day}_v04.cdf')
            bimax = cdflib.xarray.cdf_to_xarray(f'./v00/{year}/{month}/spp_swp_spi_sf00_fits_{date}_v00.cdf', fillval_to_nan=True)
            valid_dates.append(date)
        except FileNotFoundError:
            print(f"File not found for date: {date}")
            continue

        spc = spc.rename({'Epoch':'time'})
        span = span.drop_vars('ROTMAT_SC_INST', errors='ignore')

        # Set all fill_vals to be nans.
        spc = spc.where(spc != -1e31, np.nan)
        span = span.where(span != -1e31, np.nan)
        bimax = bimax.where(bimax != -1e31, np.nan)

        # Interpolate the span data to the SPC time points
        span_interp = span.interp(Epoch=spc.time.data)
        bimax_interp = bimax.interp(Epoch=spc.time.data)
        
        # Extract the temperatures to see if the values agree.
        T_spc  = ((0.5 * c.m_p * (spc.wp1_fit.data * u.km/u.s)**2) / c.k_B).to('K')
        T_span = ((span_interp.TEMP.data * u.eV)/c.k_B).to('K')

        # Define the ratio of the two temperatures
        T_ratio = T_spc / T_span

        # Get the Partial Anisotropy from the SPAN data
        T_fa = Partial_T_anisotropy(span_interp.T_TENSOR_INST.data[:,0], span_interp.T_TENSOR_INST.data[:,2], span_interp.MAGF_INST.data)

        Tperp_span, Tpara_span = T_fa
        Tani_span = Tperp_span/Tpara_span

        # From the bimax files we will alawys assume the core has a higher density.
        den1 = bimax_interp.np1.values
        den2 = bimax_interp.np2.values

        Trat1 = bimax_interp.Trat1
        Trat2 = bimax_interp.Trat2

        T_bimax_fits = np.zeros(len(Trat1))

        T_bimax_fits[den1 > den2] = Trat1[den1 > den2]
        T_bimax_fits[den2 > den1] = Trat2[den2 > den1] 

        # Get the span elevation and azimuthal flow angles. 
        el, az = flow_angles_span_inst(span_interp.VEL_INST.data)
        deg_thres = 2.0

        az_flag = np.abs(spc.peak_azimuth - az) < deg_thres
        el_flag = np.abs(spc.peak_elevation - el) < deg_thres
        angle_flag = az_flag & el_flag

        # Choose only the uncertainty flags that are relevant
        upper_percent_ani = spc.T_ani_avg_delta_plus.data / spc.T_ani_avg.data
        lower_percent_ani = spc.T_ani_avg_delta_minus.data / spc.T_ani_avg.data
        uncertainty_flag = (upper_percent_ani < 0.2) & (lower_percent_ani < 0.2)

        # Get the max flag
        max_flag = (updated_flags2[date].total_max_flag > 0) 
        spc_valid_flag = max_flag & (spc.T_ani_avg.data > 0.2) & (spc.T_ani_avg.data < 9.0) 
        span_valid_flag = (T_bimax_fits > 0.2) & (T_bimax_fits < 9.0) & uncertainty_flag

        # Get the masks where the temperatures agree.
        both_valid_mask = angle_flag & np.isfinite(T_span) & np.isfinite(T_spc) & spc_valid_flag & span_valid_flag

        T_ani_spc = spc.T_ani_avg.data
        T_ani_span = Tani_span

        T_ani_spc_dict.append(T_ani_spc[both_valid_mask])
        T_ani_span_dict.append(T_ani_span[both_valid_mask])

        if bimax_interp is not None:
            times_dict.append(spc.time.data[both_valid_mask])
            T_ani_spc_bimax_dict.append(T_ani_spc[both_valid_mask])
            T_ani_span_bimax_dict.append(T_ani_span[both_valid_mask])
            T_ani_bimax_dict.append(T_bimax_fits[both_valid_mask])

    if not T_ani_spc_bimax_dict or not T_ani_bimax_dict:
        print("No valid events found to plot.")
    else:
        # Concatenate unlogged arrays
        A = np.concatenate(T_ani_spc_bimax_dict)
        B = np.concatenate(T_ani_bimax_dict)

        # Filter for finite, strictly positive data
        valid_mask = np.isfinite(A) & np.isfinite(B) & (A > 0) & (B > 0)
        x = A[valid_mask]
        y = B[valid_mask]

        
        if x.size == 0:
            print("No positive, finite data points available to display.")
        else:
            from scipy.stats import pearsonr, spearmanr
            from scipy.odr import Model, Data, ODR

            # 1. Compute Log Statistics
            log_x = np.log10(x)
            log_y = np.log10(y)
            
            r_log, _ = pearsonr(log_x, log_y)
            rho, _   = spearmanr(log_x, log_y)
            delta_dex = log_y - log_x
            med_bias  = np.median(delta_dex)
            std_dex   = np.std(delta_dex)

            # 2. Compute Running Medians
            bin_edges = np.logspace(np.log10(0.2), np.log10(8.0), 12)
            bin_centers, medians, p16s, p84s = [], [], [], []
            for i in range(len(bin_edges) - 1):
                mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
                if np.sum(mask) >= 10:
                    bin_centers.append(np.sqrt(bin_edges[i] * bin_edges[i+1]))  # Geometric center
                    y_slice = y[mask]
                    medians.append(np.median(y_slice))
                    p16s.append(np.percentile(y_slice, 16))
                    p84s.append(np.percentile(y_slice, 84))

            # 3. Plotting
            val_min, val_max = 0.1, 10.0
            num_bins = 45
            log_bins = np.geomspace(val_min, val_max, num_bins + 1)

            fig, ax = plt.subplots(figsize=(7, 6))

            # 2D histogram
            counts, xedges, yedges, im = ax.hist2d(
                x, y,
                bins=[log_bins, log_bins],
                norm=colors.LogNorm(),
                cmap=cmc.oslo_r
            )

            # Set axes to logarithmic scale
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(val_min, val_max)
            ax.set_ylim(val_min, val_max)

            # 1:1 reference line
            ax.plot([val_min, val_max], [val_min, val_max], 'r--', lw=1.5, label='1 : 1 Line')

            # Inset Statistics Box
            stats_str = (f"$N = {len(x):,}$\n"
                         f"$r_{{\\log}} = {r_log:.3f}$\n"
                         f"$\\rho = {rho:.3f}$\n"
                         f"$\\Delta = {med_bias:+.3f} \\pm {std_dex:.3f}$")
            ax.text(0.04, 0.95, stats_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

            # Colorbar and labels
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label('Counts per bin', fontweight='bold', fontsize=12)
            
            # Formatting (removing the title, cleaning up grid, adding specific labels)
            ax.set_xlabel(r'SPC Anisotropy ($T_\perp / T_\parallel$)', fontweight='bold', fontsize=12)
            ax.set_ylabel(r'SPAN-i Bi-Maxwellian Anisotropy ($T_\perp / T_\parallel$)', fontweight='bold', fontsize=12)
            ax.grid(True, which='both', ls=':', alpha=0.3)
            
            # Position legend below the stats box
            ax.legend(loc='lower right', framealpha=0.85)

            plt.tight_layout()
            plt.savefig('temperature_figure_updated.png', dpi=300)
            plt.show()