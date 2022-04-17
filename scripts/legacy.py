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
