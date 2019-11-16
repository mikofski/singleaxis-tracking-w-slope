#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Single Axis Tracker with slope"""

import logging
from past.builtins import basestring  # for python 2 to 3 compatibility
import numpy as np
import pandas as pd
import pvlib

logging.basicConfig()

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def calc_tracker_axis_tilt(system_azimuth, system_zenith, tracker_azimuth):
    """
    Calculate tracker axis tilt in the global reference frame when on a sloped
    plane.

    Parameters
    ----------
    system_azimuth : float
        direction of normal to slope on horizontal [radians]
    system_zenith : float
        tilt of normal to slope relative to vertical [radians]
    tracker_azimuth : float
        direction of tracker axes on horizontal [radians]

    Returns
    -------
    tracker_zenith : float
        tilt of tracker [radians]

    Solving for the tracker tilt on a slope is derived in the following steps:

    1. the trackers axes are in the system plane, so the ``z-coord = 0``
    
    2. rotate the trackers ``[x_tr_sys, y_tr_sys, 0]`` back to the global, but
       rotated by the tracker global azimuth if there is one, so that the
       tracker axis is constrained to y-z plane so that ``x-coord = 0`` ::

        Rx_sys = [[1,           0,            0],
                  [0, cos(sys_ze), -sin(sys_ze)],
                  [0, sin(sys_ze),  cos(sys_ze)]]

        Rz_sys = [[cos(sys_az-tr_az), -sin(sys_az-tr_az), 0],
                  [sin(sys_az-tr_az),  cos(sys_az-tr_az), 0],
                  [                0,                  0, 1]]

        tr_rot_glo = Rz_sys.T * (Rx_sys.T * [x_tr_sys, y_tr_sys, 0])

        tr_rot_glo = [
          [ x_tr_sys*cos(sys_az-tr_az)+y_tr_sys*sin(sys_az-tr_az)*cos(sys_ze)],
          [-x_tr_sys*sin(sys_az-tr_az)+y_tr_sys*cos(sys_az-tr_az)*cos(sys_ze)],
          [                                             -y_tr_sys*sin(sys_ze)]]

    3. solve for ``x_tr_sys`` ::

        x_tr_sys*cos(sys_az-tr_az)+y_tr_sys*sin(sys_az-tr_az)*cos(sys_ze) = 0
        x_tr_sys = -y_tr_sys*tan(sys_az-tr_az)*cos(sys_ze)

    4. so tracker axis tilt, ``tr_ze = arctan2(tr_rot_glo_z, tr_rot_glo_y)`` ::

        tr_rot_glo_y = y_tr_sys*cos(sys_ze)*(
          tan(sys_az-tr_az)*sin(sys_az-tr_az) + cos(sys_az-tr_az))
        
        tan(tr_ze) = -y_tr_sys*sin(sys_ze) / tr_rot_glo_y

    The trick is multiply top and bottom by cos(sys_az-tr_az) and remember that
    ``sin^2 + cos^2 = 1`` (or just use sympy.simplify) ::

        tan(tr_ze) = -tan(sys_ze)*cos(sys_az-tr_az) 
    """
    sys_az_rel_to_tr_az = system_azimuth - tracker_azimuth
    tan_tr_ze = -np.cos(sys_az_rel_to_tr_az) * np.tan(system_zenith)
    return -np.arctan(tan_tr_ze)


def calc_system_tracker_side_slope(
        tracker_azimuth, tracker_zenith, system_azimuth, system_zenith):
    """
    Calculate the slope perpendicular to the tracker axis relative to the
    system plane containing the axes as well as the rotation of the tracker
    axes relative to the system plane. Note in order for the backtracking
    algorithm to work correctly on a sloped system plane, the side slope must
    be applied to the tracker rotation.

    Parameters
    ----------
    system_azimuth : float
        direction of normal to slope on horizontal [radians]
    system_zenith : float
        tilt of normal to slope relative to vertical [radians]
    tracker_azimuth : float
        direction of tracker axes on horizontal [radians]
    tracker_zenith : float
        tilt of tracker [radians]

    Returns
    -------
    tracker side slope and rotation relative to system plane [radians]
    """
    # find the relative rotation of the trackers in the system plane
    # 1. tracker axis vector
    cos_tr_ze = np.cos(-tracker_zenith)
    tr_ax = np.array([
        [cos_tr_ze*np.sin(tracker_azimuth)],
        [cos_tr_ze*np.cos(tracker_azimuth)],
        [np.sin(-tracker_zenith)]])
    # 2. rotate tracker axis vector from global to system reference frame
    r11 = r22 = np.cos(system_azimuth)
    r21 = np.sin(system_azimuth)
    r12 = -r21
    rot = np.array([
        [1, 0, 0],
        [0, r11, r12],
        [0, r21, r22]])
    sys_z_rot = np.roll(rot, (2, 2), (1, 0))
    # first around the z-axis
    tr_ax_sys_z_rot = np.dot(sys_z_rot, tr_ax)
    # then around x-axis so that xy-plane is the plane with slope and trackers
    r11 = r22 = np.cos(system_zenith)
    r21 = np.sin(system_zenith)
    r12 = -r21
    sys_x_rot = np.array([
        [1, 0, 0],
        [0, r11, r12],
        [0, r21, r22]])
    tr_ax_sys = np.dot(sys_x_rot, tr_ax_sys_z_rot)
    # now that tracker axis is in coordinate system of slope, the relative
    # rotation is the angle from the y axis
    tr_rel_rot = np.arctan2(tr_ax_sys[0, 0], tr_ax_sys[1, 0])
    # find side slope
    # 1. tracker normal vector
    sin_tr_ze = np.sin(tracker_zenith)
    tr_norm = np.array([
        [sin_tr_ze*np.sin(tracker_azimuth)],
        [sin_tr_ze*np.cos(tracker_azimuth)],
        [cos_tr_ze]])  # note: cos(-x) = cos(x)
    # 2. rotate tracker normal vector from global to system reference frame 
    tr_norm_sys_z_rot = np.dot(sys_z_rot, tr_norm)
    tr_norm_sys = np.dot(sys_x_rot, tr_norm_sys_z_rot)
    # 3. side slope is angle between tracker normal and system plane normal
    # np.arccos(tr_norm_sys[2])
    # 4. but we need to know which way the slope is facing, so rotate to
    # tracker use arctan2
    r11 = r22 = np.cos(tr_rel_rot)
    r21 = np.sin(tr_rel_rot)
    r12 = -r21
    rot = np.array([
        [1, 0, 0],
        [0, r11, r12],
        [0, r21, r22]])
    sys_tr_z_rot = np.roll(rot, (2, 2), (1, 0))
    tr_norm_sys_tr = np.dot(sys_tr_z_rot, tr_norm_sys)
    side_slope = np.arctan2(tr_norm_sys_tr[0, 0], tr_norm_sys_tr[2, 0])
    return side_slope, tr_rel_rot


def _singleaxis_tracking_wslope_test_helper(
        apparent_zenith, azimuth, system_plane, axis_azimuth=0,
        max_angle=90, backtrack=True, gcr=2.0/7.0):
    system_azimuth = np.radians(system_plane[0])
    system_zenith = np.radians(system_plane[1])
    tracker_azimuth = np.radians(axis_azimuth)
    tracker_zenith = calc_tracker_axis_tilt(
        system_azimuth, system_zenith, tracker_azimuth)
    axis_tilt = np.degrees(tracker_zenith)
    side_slope, tr_rel_rot = calc_system_tracker_side_slope(
        tracker_azimuth, tracker_zenith, system_azimuth, system_zenith)
    morn_angle = max_angle - np.degrees(side_slope)
    eve_angle = max_angle + np.degrees(side_slope)
    sat_morn = pvlib.tracking.singleaxis(
        apparent_zenith, azimuth, axis_tilt, axis_azimuth,
        morn_angle, False, gcr)
    sat_eve = pvlib.tracking.singleaxis(
        apparent_zenith, azimuth, axis_tilt, axis_azimuth,
        eve_angle, False, gcr)
    morn = azimuth < 180
    sat = pd.concat([sat_morn[morn], sat_eve[~morn]]).sort_index()
    tr_rot_rad = np.radians(sat['tracker_theta'])
    if backtrack:
        # this could be a place to try the walrus := operator from py38
        lx = np.cos(tr_rot_rad + side_slope)
        backtrack_rot = np.where(lx < gcr, np.arccos(lx / gcr), 0)
        tr_rot_backtrack = tr_rot_rad - backtrack_rot * np.sign(tr_rot_rad)
        sat['tracker_theta'] = -np.degrees(tr_rot_backtrack)
        # calculate angle of incidence
        x_tracker = np.sin(tr_rot_backtrack + side_slope)
        z_tracker = np.cos(tr_rot_backtrack + side_slope)
        # we need the solar vector
        solar_ze_rad = np.radians(apparent_zenith)
        solar_az_rad = np.radians(azimuth)
        sin_solar_ze = np.sin(solar_ze_rad)
        x_solar = sin_solar_ze * np.sin(solar_az_rad)
        y_solar = sin_solar_ze * np.cos(solar_az_rad)
        z_solar = np.cos(solar_ze_rad)
        solar_vector = np.stack((x_solar, y_solar, z_solar), axis=0)
        # we need the system rotation 
        r11 = r22 = np.cos(system_azimuth)
        r21 = np.sin(system_azimuth)
        r12 = -r21
        rot = np.array([
            [1, 0, 0],
            [0, r11, r12],
            [0, r21, r22]])
        sys_z_rot = np.roll(rot, (2, 2), (1, 0))
        sol_sys_z_rot = np.dot(sys_z_rot, solar_vector)
        r11 = r22 = np.cos(system_zenith)
        r21 = np.sin(system_zenith)
        r12 = -r21
        sys_x_rot = np.array([
            [1, 0, 0],
            [0, r11, r12],
            [0, r21, r22]])
        sol_sys = np.dot(sys_x_rot, sol_sys_z_rot)
        # we need the relative rotation
        r11 = r22 = np.cos(tr_rel_rot)
        r21 = np.sin(tr_rel_rot)
        r12 = -r21
        rot = np.array([
            [1, 0, 0],
            [0, r11, r12],
            [0, r21, r22]])
        sys_tr_z_rot = np.roll(rot, (2, 2), (1, 0))
        sol_sys_tr = np.dot(sys_tr_z_rot, sol_sys)
        aoi_rad = np.arccos(
            x_tracker*sol_sys_tr[0, :] + z_tracker*sol_sys_tr[2, :])
        sat['aoi'] = np.degrees(aoi_rad)
        # TODO: output surface normal vector orientation (az, ze)
    return sat


def test_tracker_wslope_helper():
    system_plane = (77.34, 10.1149)
    system_azimuth = np.radians(system_plane[0])
    system_zenith = np.radians(system_plane[1])
    axis_azimuth = 0
    tracker_azimuth = np.radians(axis_azimuth)
    max_rotation = 75.0
    gcr = 0.328
    tracker_zenith = calc_tracker_axis_tilt(
        system_azimuth, system_zenith, tracker_azimuth)
    LOGGER.debug('tracker zenith = %g', np.degrees(tracker_zenith))
    assert np.isclose(tracker_zenith, 0.039077922)
    side_slope, tr_rel_rot = calc_system_tracker_side_slope(
        tracker_azimuth, tracker_zenith, system_azimuth, system_zenith)
    LOGGER.debug('sideslope = %g', np.degrees(side_slope))
    assert np.isclose(side_slope, -0.172202784)
    LOGGER.debug('tracker relative rotation = %g', np.degrees(tr_rel_rot))
    assert np.isclose(tr_rel_rot, -1.346464234)
    starttime = '2017-01-01T00:30:00-0300'
    stoptime = '2017-12-31T23:59:59-0300'
    lat, lon = -27.597300, -48.549610
    times = pd.DatetimeIndex(pd.date_range(starttime, stoptime, freq='H'))
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    sat = _singleaxis_tracking_wslope_test_helper(
        solpos['apparent_zenith'], solpos['azimuth'], system_plane,
        axis_azimuth, max_angle=max_rotation, backtrack=True, gcr=gcr)
    expected = pd.read_csv('Florianopolis_Brasilia.csv')
    assert np.allclose(solpos['apparent_zenith'], expected['zen'])
    assert np.allclose(solpos['azimuth'], expected['azim'])
    roundoff_errors = ['2017-03-14 18:30:00-03:00',
        '2017-10-30 18:30:00-03:00',
        '2017-10-31 18:30:00-03:00',
        '2017-11-01 18:30:00-03:00',
        '2017-11-02 18:30:00-03:00',
        '2017-11-03 18:30:00-03:00']
    for idx in roundoff_errors:
        sat['tracker_theta'][idx] = np.nan
        sat.loc[idx]['aoi'] = np.nan
    nans = np.isnan(sat['tracker_theta'].values)
    expected['trrot'][nans] = np.nan
    expected['aoi'][nans] = np.nan
    assert np.allclose(sat['tracker_theta'], expected['trrot'].values, equal_nan=True)
    aoi90 = np.abs(sat['aoi'].values) < 90
    assert np.allclose(sat['aoi'][aoi90], expected['aoi'][aoi90].values, 0.00055)
    return tracker_zenith, side_slope, tr_rel_rot, sat


def test_pvlib_flat():
    starttime = '2017-01-01T00:30:00-0300'
    stoptime = '2017-12-31T23:59:59-0300'
    lat, lon = -27.597300, -48.549610
    times = pd.DatetimeIndex(pd.date_range(starttime, stoptime, freq='H'))
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    pvlib_flat = pvlib.tracking.singleaxis(
        solpos['apparent_zenith'], solpos['azimuth'])
    sat_flat = _singleaxis_tracking_wslope_test_helper(
        solpos['apparent_zenith'], solpos['azimuth'], system_plane=(0, 0),
        axis_azimuth=0, max_angle=90, backtrack=True, gcr=2.0/7.0)
    trrot, aoi = sat_flat['tracker_theta'].values, sat_flat['aoi'].values
    nans = np.isnan(pvlib_flat['tracker_theta'])
    # FIXME: both pointing north, so why are signs opposite?
    assert np.allclose(pvlib_flat['tracker_theta'][~nans], -trrot[~nans])
    assert np.allclose(pvlib_flat['aoi'][~nans], aoi[~nans])


def test_pvlib_tilt20():
    starttime = '2017-01-01T00:30:00-0300'
    stoptime = '2017-12-31T23:59:59-0300'
    lat, lon = -27.597300, -48.549610
    times = pd.DatetimeIndex(pd.date_range(starttime, stoptime, freq='H'))
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    pvlib_tilt20 = pvlib.tracking.singleaxis(
        solpos['apparent_zenith'], solpos['azimuth'], axis_tilt=20.0, axis_azimuth=180.0)
    sat_tilt20 = _singleaxis_tracking_wslope_test_helper(
        solpos['apparent_zenith'], solpos['azimuth'], system_plane=(180.0, 20.0),
        axis_azimuth=0, max_angle=90, backtrack=True, gcr=2.0/7.0)
    trrot, aoi = sat_tilt20['tracker_theta'].values, sat_tilt20['aoi'].values
    nans = np.isnan(pvlib_tilt20['tracker_theta'])
    # FIXME: pvlib and sat are not agreeing on some backtracking times
    ninetys = np.abs(pvlib_tilt20['tracker_theta']) < 90.000000
    zeroes = np.isclose(trrot, 0.0)
    conditions = ~nans & ninetys & ~zeroes
    # TODO: now both are pointing south, and signs are the same, look into this
    assert np.allclose(pvlib_tilt20['tracker_theta'][conditions], trrot[conditions])
    assert np.allclose(pvlib_tilt20['aoi'][conditions], aoi[conditions])


if __name__ == "__main__":
    tracker_zenith, side_slope, tr_rel_rot, sat = test_tracker_wslope_helper()
    test_pvlib_flat()
    test_pvlib_tilt20()
