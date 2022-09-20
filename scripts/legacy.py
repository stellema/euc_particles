# -*- coding: utf-8 -*-
"""Legacy code.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sun Apr 17 14:52:59 2022

"""


def update_formatted_file_sources(lon, exp, v, r):
    """Reapply source locations found for post-formatting file.

    This function only needs to run for files formatted using an old version of
    source updater.

    Assumes:
        - files to update in data/plx/tmp/
        - Updated files in data/plx/ (won't run if already found here).

    Todo:
        - Fix traj indexing between old/formatted files.
    """
    import numpy as np
    import xarray as xr
    from tools import save_dataset
    from fncs import (get_plx_id, update_particle_data_sources,
                      get_index_of_last_obs)

    xid = get_plx_id(exp, lon, v, r, 'plx/tmp')
    xid_new = get_plx_id(exp, lon, v, r, 'plx')

    # Check if file already updated.
    if xid_new.exists():
        return

    ds_full = xr.open_dataset(xid, chunks='auto')

    # Apply updates to ds & subset back into full only if needed.
    ds = ds_full.copy()

    # Expand variable to 2D (all zeros).
    ds['zone'] = ds.zone.broadcast_like(ds.age).copy()
    ds['zone'] *= 0

    # Reapply source definition fix.
    ds = update_particle_data_sources(ds)

    # Find which particles need to be updated.
    # Check any zones are reached earlier than in original data.
    obs_old = get_index_of_last_obs(ds_full, np.isnan(ds_full.age))
    obs_new = get_index_of_last_obs(ds, ds.zone > 0.)

    # Traj location indexes.
    traj_to_replace = ds_full.traj[obs_new < obs_old].traj
    traj_to_replace = ds_full.indexes['traj'].get_indexer(traj_to_replace)

    # Subset the particles that need updating.
    ds = ds.isel(traj=traj_to_replace)

    # Reapply mask that cuts off data after particle reaches source.
    ds = ds.where(ds.obs <= obs_new)

    # Change zone back to 1D (last found).
    ds['zone'] = ds.zone.max('obs')

    # Replace the modified subset back into full dataset.
    for var in ds_full.data_vars:
        ds_full[dict(traj=traj_to_replace)][var] = ds[var]

    # Re-save.
    msg = ': Updated source definitions.'
    save_dataset(ds_full, xid_new, msg)
    return


def combined_source_histogram(ds, lon):
    """Histograms of source variables plot."""

    def plot_histogram(ax, dx, var, color, cutoff=0.85, weighted=True, name=''):
        """Plot histogram with historical (solid) & projection (dashed)."""

        kwargs = dict(histtype='step', density=False, range=tuple(cutoff),
                      stacked=False, alpha=1, cumulative=False, color=color,
                      edgecolor=color, hatch=None, lw=1.4, label=name)
        dx = [dx.isel(exp=i).dropna('traj', 'all') for i in [0, 1]]
        bins = 'fd'
        weights = None
        if weighted:
            # weights = [dx[i].u / dx[i].u.sum().item() for i in [0, 1]]
            weights = [dx[i].u for i in [0, 1]]
            # weights = [dx[i].u / dx[i].u_zone.mean().item() for i in [0, 1]]

            # Find number of bins based on combined hist/proj data range.
            h0, _, r0 = weighted_bins_fd(dx[0][var], weights[0])
            h1, _, r1 = weighted_bins_fd(dx[1][var], weights[1])

            # Data min & max of both datasets.
            r = [min(np.floor([r0[0], r1[0]])), max(np.ceil([r0[1], r1[1]]))]
            kwargs['range'] = r

            # Number of bins for combined data range (use smallest bin width).
            bins = int(np.ceil(np.diff(r) / min([h0, h1])))

        # Historical.
        x, _bins, _ = ax.hist(dx[0][var], bins, weights=weights[0], **kwargs)

        # # RCP8.5.
        # kwargs.update(dict(ls='--'))
        # bins = bins if weighted else _bins
        # _, bins, _ = ax.hist(dx[1][var], bins, weights=weights[1], **kwargs)

        ax.set_xlim(xmin=cutoff[0], xmax=cutoff[1])
        return ax

    zn = ds.zone.values[:-2]
    varz = ['age', 'distance', 'speed']
    fig, axes = plt.subplots(len(varz), 1, figsize=(10, 15))
    i = 0
    for vi, var in enumerate(varz):
        ax = axes.flatten()[i]
        cutoff = [[0, 1500], [0, 30], [0.06, 0.35]][vi]  # xaxis limits.
        name, units = ds[var].attrs['long_name'], ds[var].attrs['units']
        ax.set_title('{} {}'.format(ltr[i], name), loc='left')

        for zi, z in enumerate(zn):
            color = ds.colors[zi].item()
            zname = ds.names[zi].item()
            dx = ds.sel(zone=z)
            ax = plot_histogram(ax, dx, var, color, cutoff=cutoff, name=zname)

        # Create new legend handles but use the colors from the existing ones
        handles, labels = ax.get_legend_handles_labels()
        handles = [mpl.lines.Line2D([], [], c=h.get_edgecolor()) for h in handles]
        ax.legend(handles=handles, labels=labels, loc='best')
        ax.set_xlabel('{} [{}]'.format(name, units))
        ax.set_ylabel('Transport [Sv]')
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        i += 1

    # Format plots.
    plt.suptitle('{}°E'.format(lon))

    plt.tight_layout()
    plt.savefig(cfg.fig / 'sources/histogram_{}_comb.png'.format(lon))
    plt.show()
    return


def timeseries_bar(exp=0, z_ids=list(range(9)), sum_interior=True):
    """Bar graph of source transport for each release longitude.

    Horizontal bar graph (sources on y-axis) with or without RCP8.5.
    4 (2x2) subplots for each release longitude.

    Args:
        exp (str, optional): Historical or with RCP8.5. Defaults to 0.
        sum_interior (bool, optional): DESCRIPTION. Defaults to True.
    # # stacked bar.
    # for z in np.arange(dx.zone.size)[::-1]:
    #     h = dx.isel(zone=z).values
    #     ax.bar(x, h, bottom=b, color=c[z], label=xlabels[z], **kwargs)
    #     b += h
    """
    i = 2
    lon = cfg.lons[i]
    ds = source_dataset(lon, sum_interior)
    ds = ds.isel(zone=z_ids)  # Single.
    # ds = merge_LLWBC_interior_sources(ds).isel(zone=[-2, -1])  # Merged.

    dx = ds.u_zone.isel(exp=exp)
    dx = dx.where(dx.rtime < np.datetime64('2013-01-06'), drop=1)
    dx = dx.resample(rtime="1m").mean("rtime", keep_attrs=True)
    dx = dx.rolling(rtime=6).mean()

    x = dx.rtime.dt.strftime("%Y-%m")
    b = np.zeros(dx.rtime.size)
    xlabels, c = ds.names.values, ds.colors.values
    inds = [-1, -2, 3, 4, 6, 5, 2, 0, 1]  # np.arange(dx.zone.size)
    xlabels, c = xlabels[inds], c[inds]
    zi = np.arange(dx.zone.size)[inds]
    # d = [dx.isel(zone=z) for z in zi]
    d = [dx.isel(zone=z) - dx.isel(zone=z).mean('rtime') for z in zi]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), sharey='row', sharex='all')
    ax.stackplot(x, *d, colors=c, labels=xlabels,
                 baseline=['zero', 'sym', 'wiggle', 'weighted_wiggle'][2])
    ax.margins(x=0, y=0)
    lgd = ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    ax.set_xticks(x[::46])
    ax.set_title('{} EUC sources at {}°E'.format(ltr[i], lon), loc='left')
    return


def source_timeseries(exp, lon, var='u_zone', merge_straits=False, anom=True):
    """Timeseries plot."""
    ds = source_dataset(lon, sum_interior=True)

    # Annual mea nbetween scenario years.
    times = slice('2012') if exp == 0 else slice('2070', '2101')
    ds = ds.sel(rtime=times).sel(exp=exp)
    dsm = ds.resample(rtime="1y").mean("rtime", keep_attrs=True)

    # Plot timeseries of source transport.
    if 'long_name' in ds[var].attrs:
        name = ds[var].attrs['long_name']
        units = ds[var].attrs['units']
    else:
        name, units = 'Transport', 'Sv'

    sourceids = [1, 2, 3, 6, 7]
    xdim = dsm.rtime
    names, colours = ds.names.values, ds.colors.values

    fig, ax = plt.subplots(1, figsize=(7, 3))

    for i, z in enumerate(sourceids):
        if merge_straits and z == 1:
            dz = dsm.u_zone.sel(zone=[1, 2]).sum('zone')
        else:
            dz = dsm.u_zone.sel(zone=z)

        if anom:
            dz = dz - dz.mean('rtime')
            ax.axhline(0, color='grey')

        ax.plot(xdim, dz, c=colours[z], label=names[z])

    ax.set_title('{} EUC {} at {}°E'.format(cfg.exps[exp], name.lower(), lon),
                 loc='left')
    ax.set_ylabel('{} [{}]'.format(name, units))
    ax.margins(x=0)

    lgd = ax.legend()
    plt.tight_layout()
    file = 'source_{}_timeseries_{}_{}_{}'.format(name, lon, cfg.exp[exp],
                                                  ''.join(map(str, sourceids)))
    if anom:
        file + '_anom'
    plt.savefig(cfg.fig / (file + '.png'), bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.show()
    return
