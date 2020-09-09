# -*- coding: utf-8 -*-
"""
created: Thu Sep  3 14:59:05 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import cfg
import tools
import copy
import math
import warnings
import cartopy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from datetime import timedelta as delta
from parcels.field import Field
from parcels.field import VectorField
from parcels.grid import GridCode
from parcels.grid import CurvilinearGrid
from parcels.tools.error import TimeExtrapolationError
from parcels.tools.loggers import logger
warnings.filterwarnings("ignore")


def plotparticles(particles, with_particles=True, show_time=None, field=None,
                  domain=None, projection=None, land=True, vmin=None,
                  vmax=None, savefile=None, animation=False, **kwargs):
    """Function to plot a Parcels ParticleSet

    :param show_time: Time at which to show the ParticleSet
    :param with_particles: Boolean whether particles are also plotted on Field
    :param field: Field to plot under particles (either None, a Field object, or 'vector')
    :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
    :param projection: type of cartopy projection to use (default PlateCarree)
    :param land: Boolean whether to show land. This is ignored for flat meshes
    :param vmin: minimum colour scale (only in single-plot mode)
    :param vmax: maximum colour scale (only in single-plot mode)
    :param savefile: Name of a file to save the plot to
    :param animation: Boolean whether result is a single plot, or an animation
    """

    show_time = particles[0].time if show_time is None else show_time
    if isinstance(show_time, datetime):
        show_time = np.datetime64(show_time)
    if isinstance(show_time, np.datetime64):
        if not particles.time_origin:
            raise NotImplementedError(
                'If fieldset.time_origin is not a date, showtime cannot be a date in particleset.show()')
        show_time = particles.time_origin.reltime(show_time)
    if isinstance(show_time, delta):
        show_time = show_time.total_seconds()
    if np.isnan(show_time):
        show_time, _ = particles.fieldset.gridset.dimrange('time_full')

    if field is None:
        spherical = True if particles.fieldset.U.grid.mesh == 'spherical' else False
        fig, ax = create_parcelsfig_axis(spherical, land, projection)

        ax.set_title('Particles' + parsetimestr(particles.fieldset.U.grid.time_origin, show_time))
        latN, latS, lonE, lonW = parsedomain(domain, particles.fieldset.U)
        if cartopy is None or projection is None:
            if domain is not None:
                if isinstance(particles.fieldset.U.grid, CurvilinearGrid):
                    ax.set_xlim(particles.fieldset.U.grid.lon[latS, lonW], particles.fieldset.U.grid.lon[latN, lonE])
                    ax.set_ylim(particles.fieldset.U.grid.lat[latS, lonW], particles.fieldset.U.grid.lat[latN, lonE])
                else:
                    ax.set_xlim(particles.fieldset.U.grid.lon[lonW], particles.fieldset.U.grid.lon[lonE])
                    ax.set_ylim(particles.fieldset.U.grid.lat[latS], particles.fieldset.U.grid.lat[latN])
            else:
                ax.set_xlim(np.nanmin(particles.fieldset.U.grid.lon), np.nanmax(particles.fieldset.U.grid.lon))
                ax.set_ylim(np.nanmin(particles.fieldset.U.grid.lat), np.nanmax(particles.fieldset.U.grid.lat))
        elif domain is not None:
            if isinstance(particles.fieldset.U.grid, CurvilinearGrid):
                ax.set_extent([particles.fieldset.U.grid.lon[latS, lonW], particles.fieldset.U.grid.lon[latN, lonE],
                               particles.fieldset.U.grid.lat[latS, lonW], particles.fieldset.U.grid.lat[latN, lonE]])
            else:
                ax.set_extent([particles.fieldset.U.grid.lon[lonW], particles.fieldset.U.grid.lon[lonE],
                               particles.fieldset.U.grid.lat[latS], particles.fieldset.U.grid.lat[latN]])

    else:
        if field == 'vector':
            field = particles.fieldset.UV
        elif not isinstance(field, Field):
            field = getattr(particles.fieldset, field)

        depth_level = kwargs.pop('depth_level', 0)
        fig, ax = plotfield(field=field, animation=animation, show_time=show_time, domain=domain,
                                          projection=projection, land=land, vmin=vmin, vmax=vmax, savefile=None,
                                          titlestr='Particles and ', depth_level=depth_level)

    if with_particles:
#############################ADDED#############################################
        if 'unbeached' in particles.particle_data:
            plon = np.array([p.lon for p in particles if p.unbeached == 0])
            plat = np.array([p.lat for p in particles if p.unbeached == 0])
            plonb = np.array([p.lon for p in particles if p.unbeached != 0])
            platb = np.array([p.lat for p in particles if p.unbeached != 0])
            c = np.array([p.unbeached for p in particles if p.unbeached != 0], dtype=int)
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, 15))
            c = np.where(c >= 15, 14, c)
        else:
            plon = np.array([p.lon for p in particles])
            plat = np.array([p.lat for p in particles])
#############################OLD##############################################
        # plon = np.array([p.lon for p in particles])
        # plat = np.array([p.lat for p in particles])
##############################################################################
        if cartopy:
            ax.scatter(plon, plat, s=2, color='black', zorder=20, transform=cartopy.crs.PlateCarree())
#############################ADDED#############################################
            if 'unbeached' in particles.particle_data:
                ax.scatter(plonb, platb, s=2, c=colors[c], zorder=20, transform=cartopy.crs.PlateCarree())
##############################################################################
        else:
            ax.scatter(plon, plat, s=20, color='black', zorder=20)

    if animation:
        plt.draw()
        plt.pause(0.0001)
    elif savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)
        logger.info('Plot saved to ' + savefile + '.png')
        plt.close()


def plotfield(field, show_time=None, domain=None, depth_level=0, projection=None, land=True,
              vmin=None, vmax=None, savefile=None, **kwargs):
    """Function to plot a Parcels Field

    :param show_time: Time at which to show the Field
    :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
    :param depth_level: depth level to be plotted (default 0)
    :param projection: type of cartopy projection to use (default PlateCarree)
    :param land: Boolean whether to show land. This is ignored for flat meshes
    :param vmin: minimum colour scale (only in single-plot mode)
    :param vmax: maximum colour scale (only in single-plot mode)
    :param savefile: Name of a file to save the plot to
    :param animation: Boolean whether result is a single plot, or an animation
    """

    if type(field) is VectorField:
        spherical = True if field.U.grid.mesh == 'spherical' else False
        field = [field.U, field.V]
        plottype = 'vector'
    elif type(field) is Field:
        spherical = True if field.grid.mesh == 'spherical' else False
        field = [field]
        plottype = 'scalar'
    else:
        raise RuntimeError('field needs to be a Field or VectorField object')

    if field[0].grid.gtype in [GridCode.CurvilinearZGrid, GridCode.CurvilinearSGrid]:
        logger.warning('Field.show() does not always correctly determine the domain for curvilinear grids. '
                       'Use plotting with caution and perhaps use domain argument as in the NEMO 3D tutorial')

    fig, ax = create_parcelsfig_axis(spherical, land, projection=projection)
    if plt is None:
        return None, None, None, None  # creating axes was not possible

    data = {}
    plotlon = {}
    plotlat = {}
    for i, fld in enumerate(field):
        show_time = fld.grid.time[0] if show_time is None else show_time
        if fld.grid.defer_load:
            fld.fieldset.computeTimeChunk(show_time, 1)
        (idx, periods) = fld.time_index(show_time)
        show_time -= periods * (fld.grid.time_full[-1] - fld.grid.time_full[0])
        # if show_time > fld.grid.time[-1] or show_time < fld.grid.time[0]:
        #     raise TimeExtrapolationError(show_time, field=fld, msg='show_time')
        latN, latS, lonE, lonW = parsedomain(domain, fld)
        if isinstance(fld.grid, CurvilinearGrid):
            plotlon[i] = fld.grid.lon[latS:latN, lonW:lonE]
            plotlat[i] = fld.grid.lat[latS:latN, lonW:lonE]
        else:
            plotlon[i] = fld.grid.lon[lonW:lonE]
            plotlat[i] = fld.grid.lat[latS:latN]
        if i > 0 and not np.allclose(plotlon[i], plotlon[0]):
            raise RuntimeError('VectorField needs to be on an A-grid for plotting')
        if fld.grid.time.size > 1:
            if fld.grid.zdim > 1:
                data[i] = np.squeeze(fld.temporal_interpolate_fullfield(idx, show_time))[depth_level, latS:latN, lonW:lonE]
            else:
                data[i] = np.squeeze(fld.temporal_interpolate_fullfield(idx, show_time))[latS:latN, lonW:lonE]
        else:
            if fld.grid.zdim > 1:
                data[i] = np.squeeze(fld.data)[depth_level, latS:latN, lonW:lonE]
            else:
                data[i] = np.squeeze(fld.data)[latS:latN, lonW:lonE]

    if plottype == 'vector':
        if field[0].interp_method == 'cgrid_velocity':
            logger.warning_once('Plotting a C-grid velocity field is achieved via an A-grid projection, reducing the plot accuracy')
            d = np.empty_like(data[0])
            d[:-1, :] = (data[0][:-1, :] + data[0][1:, :]) / 2.
            d[-1, :] = data[0][-1, :]
            data[0] = d
            d = np.empty_like(data[0])
            d[:, :-1] = (data[0][:, :-1] + data[0][:, 1:]) / 2.
            d[:, -1] = data[0][:, -1]
            data[1] = d

        spd = data[0] ** 2 + data[1] ** 2
        speed = np.where(spd > 0, np.sqrt(spd), 0)
        vmin = speed.min() if vmin is None else vmin
        vmax = speed.max() if vmax is None else vmax
        if isinstance(field[0].grid, CurvilinearGrid):
            x, y = plotlon[0], plotlat[0]
        else:
            x, y = np.meshgrid(plotlon[0], plotlat[0])
        u = np.where(speed > 0., data[0]/speed, 0)
        v = np.where(speed > 0., data[1]/speed, 0)
        if cartopy:
            cs = ax.quiver(np.asarray(x), np.asarray(y), np.asarray(u), np.asarray(v), speed, cmap=plt.cm.gist_ncar, clim=[vmin, vmax], scale=50, transform=cartopy.crs.PlateCarree())
        else:
            cs = ax.quiver(x, y, u, v, speed, cmap=plt.cm.gist_ncar, clim=[vmin, vmax], scale=50)
    else:
        vmin = data[0].min() if vmin is None else vmin
        vmax = data[0].max() if vmax is None else vmax
        assert len(data[0].shape) == 2
        if field[0].interp_method == 'cgrid_tracer':
            d = data[0][1:, 1:]
        elif field[0].interp_method == 'cgrid_velocity':
            if field[0].fieldtype == 'U':
                d = np.empty_like(data[0])
                d[:-1, :-1] = (data[0][1:, :-1] + data[0][1:, 1:]) / 2.
            elif field[0].fieldtype == 'V':
                d = np.empty_like(data[0])
                d[:-1, :-1] = (data[0][:-1, 1:] + data[0][1:, 1:]) / 2.
            else:  # W
                d = data[0][1:, 1:]
        else:  # if A-grid
            d = (data[0][:-1, :-1] + data[0][1:, :-1] + data[0][:-1, 1:] + data[0][1:, 1:])/4.
            d = np.where(data[0][:-1, :-1] == 0, 0, d)
            d = np.where(data[0][1:, :-1] == 0, 0, d)
            d = np.where(data[0][1:, 1:] == 0, 0, d)
            d = np.where(data[0][:-1, 1:] == 0, 0, d)
        if cartopy:
##############################################################################
            # cs = ax.pcolormesh(plotlon[0], plotlat[0], d, transform=cartopy.crs.PlateCarree())
##############################################################################
            if type(field) is not VectorField and field[0].name == 'Land':
                cs = ax.pcolormesh(plotlon[0], plotlat[0], d, cmap=plt.cm.Greys, transform=cartopy.crs.PlateCarree())
            else:
                cs = ax.pcolormesh(plotlon[0], plotlat[0], d, transform=cartopy.crs.PlateCarree())
##############################################################################
        else:
            cs = ax.pcolormesh(plotlon[0], plotlat[0], d)
    if cartopy is None:
        ax.set_xlim(np.nanmin(plotlon[0]), np.nanmax(plotlon[0]))
        ax.set_ylim(np.nanmin(plotlat[0]), np.nanmax(plotlat[0]))
    elif domain is not None:
        ax.set_extent([np.nanmin(plotlon[0]), np.nanmax(plotlon[0]), np.nanmin(plotlat[0]), np.nanmax(plotlat[0])], crs=cartopy.crs.PlateCarree())
    cs.cmap.set_over('k')
    cs.cmap.set_under('w')
    cs.set_clim(vmin, vmax)
##############################################################################
    if type(field) is VectorField and field[0].name != 'Land':
        cartopy_colorbar(cs, plt, fig, ax)
##############################################################################
    # cartopy_colorbar(cs, plt, fig, ax)
##############################################################################
    timestr = parsetimestr(field[0].grid.time_origin, show_time)
    titlestr = kwargs.pop('titlestr', '')
    if field[0].grid.zdim > 1:
        if field[0].grid.gtype in [GridCode.CurvilinearZGrid, GridCode.RectilinearZGrid]:
            gphrase = 'depth'
            depth_or_level = round(field[0].grid.depth[depth_level], 0)
        else:
            gphrase = 'level'
            depth_or_level = depth_level
        depthstr = ' at %s %g ' % (gphrase, depth_or_level)
    else:
        depthstr = ''
    if plottype == 'vector':
        ax.set_title(titlestr + 'vector field' + depthstr + timestr)
    else:
        ax.set_title(titlestr + field[0].name + depthstr + timestr)

    if not spherical:
        ax.set_xlabel('Zonal distance [m]')
        ax.set_ylabel('Meridional distance [m]')

    plt.draw()

    if savefile:
        plt.tight_layout(fig)
        plt.savefig(savefile, bbox_inches='tight')
        logger.info('Plot saved to ' + savefile + '.png')
        plt.close()

    return fig, ax


def create_parcelsfig_axis(spherical, land=True, projection=None, central_longitude=0):
    try:
        import matplotlib.pyplot as plt
    except:
        logger.info("Visualisation is not possible. Matplotlib not found.")
        return None, None, None, None  # creating axes was not possible

    if projection is not None and not spherical:
        raise RuntimeError('projection not accepted when Field doesn''t have geographic coordinates')

    if spherical:
        try:
            import cartopy
        except:
            logger.info("Visualisation of field with geographic coordinates is not possible. Cartopy not found.")
            return None, None, None, None  # creating axes was not possible

        projection = cartopy.crs.PlateCarree(central_longitude) if projection is None else projection
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': projection})
        try:  # gridlines not supported for all projections
            gl = ax.gridlines(crs=projection, draw_labels=True)
            gl.xlabels_top, gl.ylabels_right = (False, False)
            gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
            gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
        except:
            pass

        if land:
            ax.coastlines()
    else:
        cartopy = None
        fig, ax = plt.subplots(1, 1)
        ax.grid()
    return fig, ax


def parsedomain(domain, field):
    field.grid.check_zonal_periodic()
    if domain is not None:
        if not isinstance(domain, dict) and len(domain) == 4:  # for backward compatibility with <v2.0.0
            domain = {'N': domain[0], 'S': domain[1], 'E': domain[2], 'W': domain[3]}
        _, _, _, lonW, latS, _ = field.search_indices(domain['W'], domain['S'], 0, 0, 0, search2D=True)
        _, _, _, lonE, latN, _ = field.search_indices(domain['E'], domain['N'], 0, 0, 0, search2D=True)
        return latN+1, latS, lonE+1, lonW
    else:
        if field.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearSGrid]:
            return field.grid.lon.shape[0], 0, field.grid.lon.shape[1], 0
        else:
            return len(field.grid.lat), 0, len(field.grid.lon), 0


def parsetimestr(time_origin, show_time):
    if time_origin.calendar is None:
        return ' after ' + str(delta(seconds=show_time)) + ' hours'
    else:
        date_str = str(time_origin.fulltime(show_time))
        return ' on ' + date_str[:10] + ' ' + date_str[11:19]


def cartopy_colorbar(cs, plt, fig, ax):
    cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
    fig.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    plt.colorbar(cs, cax=cbar_ax, shrink=0.9)

    def resize_colorbar(event):
        plt.draw()
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0, 0.04, posn.height])

    fig.canvas.mpl_connect('resize_event', resize_colorbar)
    resize_colorbar(None)



def particlesvid(particles, with_particles=True, show_time=None, field=None,
                  domain=None, projection=None, land=True, vmin=None,
                  vmax=None, savefile=None, animation=False, unbeach=True, **kwargs):
    """Function to plot a Parcels ParticleSet

    :param show_time: Time at which to show the ParticleSet
    :param with_particles: Boolean whether particles are also plotted on Field
    :param field: Field to plot under particles (either None, a Field object, or 'vector')
    :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
    :param projection: type of cartopy projection to use (default PlateCarree)
    :param land: Boolean whether to show land. This is ignored for flat meshes
    :param vmin: minimum colour scale (only in single-plot mode)
    :param vmax: maximum colour scale (only in single-plot mode)
    :param savefile: Name of a file to save the plot to
    :param animation: Boolean whether result is a single plot, or an animation
    """
    import matplotlib.animation as animation

    show_time = particles[0].time if show_time is None else show_time
    if isinstance(show_time, datetime):
        show_time = np.datetime64(show_time)
    if isinstance(show_time, np.datetime64):
        show_time = particles.time_origin.reltime(show_time)
    if isinstance(show_time, delta):
        show_time = show_time.total_seconds()
    if np.isnan(show_time):
        show_time, _ = particles.fieldset.gridset.dimrange('time_full')

    if field == 'vector':
        field = particles.fieldset.UV
    elif not isinstance(field, Field):
        field = getattr(particles.fieldset, field)

    depth_level = kwargs.pop('depth_level', 0)
    fig, ax = plotfield(field=field, animation=animation, show_time=show_time,
                        domain=domain, projection=projection, land=land,
                        vmin=vmin, vmax=vmax, savefile=None,
                        titlestr='Particles and ', depth_level=depth_level)

    if with_particles:
        if unbeach and 'unbeached' in particles.particle_data:
            # Particles that havent been unbeached.
            plon = np.array([p.lon for p in particles if p.unbeached == 0])
            plat = np.array([p.lat for p in particles if p.unbeached == 0])
            # Particles that have been unbeached.
            plonb = np.array([p.lon for p in particles if p.unbeached != 0])
            platb = np.array([p.lat for p in particles if p.unbeached != 0])
            # Cycle through cmap for times unbeached.
            c = np.array([p.unbeached for p in particles if p.unbeached != 0],
                         dtype=int)
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, 20))
            c = np.where(c >= 15, 20, c)
        else:
            plon = np.array([p.lon for p in particles])
            plat = np.array([p.lat for p in particles])
        if cartopy:
            ax.scatter(plon, plat, s=2, color='black', zorder=20,
                       transform=cartopy.crs.PlateCarree())
            if unbeach and 'unbeached' in particles.particle_data:
                ax.scatter(plonb, platb, s=2, c=colors[c], zorder=20,
                           transform=cartopy.crs.PlateCarree())
        else:
            ax.scatter(plon, plat, s=20, color='black', zorder=20)

    if animation:
        plt.draw()
        plt.pause(0.0001)
    elif savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)
        logger.info('Plot saved to ' + savefile + '.png')
        plt.close()



def plot3D(sim_id, ds=None):
    """Plot 3D figure of particle trajectories over time."""
    import matplotlib.ticker as ticker
    if not ds:
        # Open ParticleFile.
        ds = xr.open_dataset(sim_id, decode_cf=True)

    N = len(ds.traj)
    x, y, z = ds.lon, ds.lat, ds.z

    fig = plt.figure(figsize=(13, 10))
    # plt.suptitle(sim_id.stem, y=0.89, x=0.23)
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ds.traj)))
    ax.set_xlim(tools.rounddown(np.nanmin(x)), tools.roundup(np.nanmax(x)))
    ax.set_ylim(tools.rounddown(np.nanmin(y)), tools.roundup(np.nanmax(y)))
    ax.set_zlim(tools.roundup(np.nanmax(z)), tools.rounddown(np.nanmin(z)))

    for i in range(N):
        ax.plot3D(x[i], y[i], z[i], color=colors[i])

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    xlabels = tools.coord_formatter(xticks, convert='lon')
    ylabels = tools.coord_formatter(yticks, convert='lat')
    zlabels = ['{:.0f}m'.format(k) for k in zticks]
    ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(xlabels))
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(ylabels))
    ax.zaxis.set_major_locator(ticker.FixedLocator(zticks))
    ax.zaxis.set_major_formatter(ticker.FixedFormatter(zlabels))
    plt.tight_layout(pad=0)

    fig.savefig(cfg.fig/('parcels/' + sim_id.stem + cfg.im_ext),
                bbox_inches='tight')
    plt.show()
    plt.close()
    ds.close()
    return


def plot3Dx(sim_id, ds=None):
    """Plot 3D figure of particle trajectories over time."""

    def setup(ax, xticks, yticks, zticks, xax='lon', yax='lat'):
        xlabels = tools.coord_formatter(xticks, convert=xax)
        ylabels = tools.coord_formatter(yticks, convert=yax)
        zlabels = ['{:.0f}m'.format(k) for k in zticks]
        ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(xlabels))
        ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(ylabels))
        ax.zaxis.set_major_locator(ticker.FixedLocator(zticks))
        ax.zaxis.set_major_formatter(ticker.FixedFormatter(zlabels))
        return ax

    if not ds:
        # Open ParticleFile.
        ds = xr.open_dataset(sim_id, decode_cf=True)

    N = len(ds.traj)
    x, y, z = ds.lon, ds.lat, ds.z
    xlim = [tools.rounddown(np.nanmin(x)), tools.roundup(np.nanmax(x))]
    ylim = [tools.rounddown(np.nanmin(y)), tools.roundup(np.nanmax(y))]
    zlim = [tools.rounddown(np.nanmin(z)), tools.roundup(np.nanmax(z))]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ds.traj)))

    # Plot figure.
    fig = plt.figure(figsize=(18, 16))
    plt.suptitle(sim_id.stem, y=0.92, x=0.1)

    ax = fig.add_subplot(221, projection='3d')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(zlim[1], zlim[0])
    for i in range(N):
        ax.plot3D(x[i], y[i], z[i], color=colors[i])
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    ax = setup(ax, xticks, yticks, zticks, xax='lon', yax='lat')

    # Reverse latitude.
    ax = fig.add_subplot(222, projection='3d')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_zlim(zlim[1], zlim[0])
    for i in range(N):
        ax.plot3D(x[i], y[i], z[i], color=colors[i])
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    ax = setup(ax, xticks, yticks, zticks, xax='lon', yax='lat')

    # Switch latitude and longitude.
    ax = fig.add_subplot(223, projection='3d')
    ax.set_ylim(xlim[0], xlim[1])
    ax.set_xlim(ylim[0], ylim[1])
    ax.set_zlim(zlim[1], zlim[0])
    for i in range(N):
        ax.plot3D(y[i], x[i], z[i], color=colors[i])
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    ax = setup(ax, xticks, yticks, zticks, xax='lat', yax='lon')

    # Reverse latitude and switch latitude and longitude.
    ax = fig.add_subplot(224, projection='3d')
    ax.set_ylim(xlim[1], xlim[0])
    ax.set_xlim(ylim[0], ylim[1])
    ax.set_zlim(zlim[1], zlim[0])
    for i in range(N):
        ax.plot3D(y[i], x[i], z[i], color=colors[i])
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    ax = setup(ax, xticks, yticks, zticks, xax='lat', yax='lon')

    plt.tight_layout(pad=0)
    fig.savefig(cfg.fig/('parcels/' + sim_id.stem + 'x' + cfg.im_ext))
    # plt.show()
    plt.close()
    ds.close()

    return


def plot_traj(sim_id, var='u', traj=None, t=None, Z=290, ds=None):
    """Plot individual trajectory (3D line and 2D scatter)."""
    if not ds:
        ds = xr.open_dataset(sim_id, decode_cf=True)
    if not traj:
        try:
            ub = np.unique(ds.where(ds.unbeached >= 1).trajectory)
            ub = ub[~np.isnan(ub)].astype(int)
            print('Number of beached=', len(ub))
            for traj in ub:
                dx = ds.where(ds.trajectory == int(traj), drop=True).isel(traj=0)
                i = np.argwhere(dx.unbeached.values >= 1)[0][0]
                print('Traj={} beached at: {:.3f}, {:.3f}, {:.2f}m, {}'
                      .format(int(traj), dx.lat[i].item(), dx.lon[i].item(), dx.z[i].item(),
                              np.datetime_as_string(dx.time[i])[:10]))

            if len(ub) == 0:
                traj = int(ds.trajectory[0, 0])
                dx = ds.where(ds.trajectory == int(traj), drop=True).isel(traj=0)

        except:
            tr = np.unique(ds.where(ds.z > Z).trajectory)[0:5]
            print(tr[~np.isnan(tr)])
            traj = int(np.nanmin(tr))
            traj = int(ds.trajectory[0, 0]) if not traj else traj
            dx = ds.where(ds.trajectory == traj, drop=True).isel(traj=0)
    else:
        try:
            dx = ds.where(ds.trajectory == int(traj), drop=True).isel(traj=0)
        except:
            dx = ds.isel(traj=traj)
    bc = [i for i in range(len(dx.unbeached.values))
          if dx.unbeached[i] > dx.unbeached[i-1]]
    zmap, norm = tools.zone_cmap()
    X = [np.round(dx.lon.min(), 1)-0.5, np.round(dx.lon.max(), 1)+0.5]
    Y = [np.round(dx.lat.min(), 1)-0.25, np.round(dx.lat.max(), 1)+0.25]

    if not t:
        t = 22
    dsv = xr.open_dataset(cfg.ofam/'ocean_{}_2012_12.nc'.format(var)).isel(Time=t)[var]

    if var in ['u', 'v']:
        var_str = 'Zonal' if var == 'u' else 'Meridional'
        vmax = 0.5
        dv = dsv.sel(xu_ocean=slice(X[0], X[1]),
                     yu_ocean=slice(Y[0], Y[1])).sel(st_ocean=Z, method='nearest')
        lat, lon, z = dv.yu_ocean, dv.xu_ocean, dv.st_ocean
    else:
        var_str = 'Vertical'
        vmax = 0.001
        dv = dsv.sel(xt_ocean=slice(X[0], X[1]),
                     yt_ocean=slice(Y[0], Y[1])).sel(sw_ocean=Z, method='nearest')
        lat, lon, z = dv.yt_ocean, dv.xt_ocean, dv.sw_ocean

    # cmap = plt.cm.get_cmap("seismic").copy()
    cmap = copy.copy(plt.cm.get_cmap("seismic"))
    cmap.set_bad('grey')
    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(221, projection='3d')
    ax.set_title(sim_id.stem + ': traj=' + str(traj))
    ax.plot3D(dx.lon, dx.lat, dx.z, color='b', marker='o', linewidth=1.5, markersize=3)
    ax.scatter3D(dx.lon[bc], dx.lat[bc], dx.z[bc], color='r', s=3, zorder=3)
    ax.set_xlim(math.floor(np.nanmin(dx.lon)), math.ceil(np.nanmax(dx.lon)))
    ax.set_ylim(math.floor(np.nanmin(dx.lat)), math.ceil(np.nanmax(dx.lat)))
    ax.set_zlim(math.ceil(np.nanmax(dx.z)), math.floor(np.nanmin(dx.z)))
    ax.set_xticklabels(tools.coord_formatter(ax.get_xticks(), convert='lon'))
    ax.set_yticklabels(tools.coord_formatter(ax.get_yticks(), convert='lat'))
    ax.set_zlabel("Depth [m]")

    ax = fig.add_subplot(222)
    ax.set_title(var_str + ' velocity at {:.1f} m'.format(z.item()))
    ax.pcolormesh(lon, lat, dv, cmap=cmap, vmax=vmax, vmin=-vmax)
    # ax.scatter(dx.lon[bc], dx.lat[bc], color='y', s=3, zorder=3)
    # ax.plot(dx.lon, dx.lat, color='k', marker='o', linewidth=1, markersize=3)
    try:
        im = ax.scatter(dx.lon, dx.lat, c=dx.zone.values, marker='o',
                        cmap=zmap, linewidth=1, s=3, norm=norm)
        plt.colorbar(im)
    except:
        im = ax.scatter(dx.lon, dx.lat, marker='o', linewidth=1, s=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


    ax = fig.add_subplot(223)
    ax.set_title(sim_id.stem + ' traj=' + str(traj))
    ax.plot(dx.lat, dx.z, color='k', marker='o', linewidth=1.7, markersize=4)
    ax.scatter(dx.lat[bc], dx.z[bc], color='r', s=3, zorder=3)
    ax.set_ylim(np.max(dx.z), np.min(dx.z))
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Depth")

    ax = fig.add_subplot(224)
    ax.set_title(sim_id.stem + ' traj=' + str(traj))
    ax.scatter(dx.lon[bc], dx.z[bc], color='r', s=4, zorder=3)
    ax.plot(dx.lon, dx.z, color='k', marker='o', linewidth=1.7, markersize=4)
    ax.set_ylim(np.max(dx.z), np.min(dx.z))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Depth")

    # plt.tight_layout()
    plt.savefig(cfg.fig/'parcels/tests/traj_{}_{}_{}.png'
                .format(sim_id.stem, traj, var))
    plt.show()

    return ds, dx


def plot_beached(sim_id, depth=None):

    ds = xr.open_dataset(sim_id, decode_cf=True)
    ds = ds.where(ds.u >= 0., drop=True)

    if depth:
        tjs = np.unique(ds.where((ds.z > depth) & (ds.unbeached > 0), drop=True).trajectory)
    else:
        tjs = np.unique(ds.where(ds.unbeached > 0, drop=True).trajectory)
    tr = tjs[~np.isnan(tjs)].astype(dtype=int)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(tr)))
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, t in enumerate(tr):
        if ~np.isnan(t):
            dx = ds.where(ds.trajectory == int(t), drop=True).isel(traj=0)
            ax.plot3D(dx.lon, dx.lat, dx.z, color=colors[i])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth [m]")
    ax.set_zlim(500, 50)
    print(tr)
    return ds, tr


def plot_ubtraj(sim_id, var='u', t=22, Z=290, ds=None):
    """Plot individual trajectory (3D line and 2D scatter)."""
    if not ds:
        ds = xr.open_dataset(sim_id, decode_cf=True)

    ub = np.unique(ds.where(ds.unbeached < 0).trajectory)
    ub = ub[~np.isnan(ub)].astype(int)
    print('Number of beached=', len(ub))
    dx = ds.isel(traj=ds.isel(obs=0).trajectory.isin(ub))

    if not t:
        t = 22
    dsv = xr.open_dataset(cfg.ofam/'ocean_{}_2012_12.nc'.format(var)).isel(Time=t)[var]

    if var in ['u', 'v']:
        var_str = 'Zonal' if var == 'u' else 'Meridional'
        vmax = 0.5
        dv = dsv.sel(st_ocean=Z, method='nearest')
        lat, lon, z = dv.yu_ocean, dv.xu_ocean, dv.st_ocean
    else:
        var_str = 'Vertical'
        vmax = 0.001
        dv = dsv.sel(sw_ocean=Z, method='nearest')
        lat, lon, z = dv.yt_ocean, dv.xt_ocean, dv.sw_ocean

    # cmap = plt.cm.get_cmap("seismic").copy()
    cmap = copy.copy(plt.cm.get_cmap("seismic"))
    cmap.set_bad('grey')
    fig = plt.figure(figsize=(12, 9))
    zmap, norm = tools.zone_cmap()
    ax = fig.add_subplot()
    ax.set_title(sim_id.stem)

    ax.set_title(var_str + ' velocity at {:.1f} m'.format(z.item()))
    ax.pcolormesh(lon, lat, dv, cmap=cmap, vmax=vmax, vmin=-vmax)
    for i in range(len(dx.traj)):
        im = ax.scatter(dx.isel(traj=i).lon, dx.isel(traj=i).lat,
                        c=dx.isel(traj=i).zone.values, marker='o',
                        cmap=zmap, linewidth=1, s=3, norm=norm)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im)

    plt.savefig(cfg.fig/'parcels/tests/{}_beached.png'
                .format(sim_id.stem))
    plt.show()

    return ds, dx


def animate_particles(i, sc, ax, particles, field, kernels, runtime, dt,
                      output_file, recovery_kernels, depth_level, domain):
    particles.execute(kernels, runtime=runtime, dt=dt, output_file=output_file,
                      verbose_progress=False, recovery=recovery_kernels)
    show_time = particles[0].time

    if field == 'vector':
        field = particles.fieldset.UV
    elif not isinstance(field, Field):
        field = getattr(particles.fieldset, field)
    field.fieldset.computeTimeChunk(show_time, 1)
    if type(field) is VectorField:
        field = [field.U, field.V]
        plottype = 'vector'
    elif type(field) is Field:
        field = [field]
        plottype = 'scalar'
    (idx, periods) = field[0].time_index(show_time)
    show_time -= periods * (field[0].grid.time_full[-1] - field[0].grid.time_full[0])
    timestr = parsetimestr(field[0].grid.time_origin, show_time)
    titlestr = 'Particles and '
    gphrase = 'depth'
    depth_or_level = round(field[0].grid.depth[depth_level], 0)
    depthstr = ' at %s %g m' % (gphrase, depth_or_level)

    if plottype == 'vector':
        ax.set_title(titlestr + 'vector field' + depthstr + timestr)
    else:
        ax.set_title(titlestr + field[0].name + depthstr + timestr)
    # Update particles.
    plon = np.array([p.lon for p in particles])
    plat = np.array([p.lat for p in particles])
    if 'unbeached' in particles.particle_data:
        c = np.array([p.unbeached for p in particles], dtype=int)
    X = np.c_[plon, plat]
    sc.set_offsets(X)
    sc.set_array(c)
    return sc,
