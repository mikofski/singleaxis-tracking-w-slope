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
AXES = ['x', 'y', 'z']
DAY = 90.0


def get_rotation_matrix(angle, axis):
    """
    Return a rotation matrix that when multiplied by a column vector returns
    a new column vector that is rotated clockwise around the given axis by the
    given angle.

    Parameters
    ----------
    angle : float
        Angle of rotation [radians]
    axis : int, str
        Axis of rotation

    Returns
    -------
    rotation matrix

    References:
       `Rotation Matrix <https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations>`_

    """
    r11 = r22 = np.cos(angle)
    r21 = np.sin(angle)
    r12 = -r21
    rot = np.array([
        [1, 0, 0],
        [0, r11, r12],
        [0, r21, r22]])
    if isinstance(axis, basestring):
        # if axis is a string, convert it to an integer
        axis = AXES.index(axis)
    while axis:
        axis -= 1
        rot = np.roll(rot, (1, 1), (1, 0))
    return rot


def get_solar_vector(solar_zenith, solar_azimuth):
    solar_ze_rad = np.radians(solar_zenith)
    solar_az_rad = np.radians(solar_azimuth)
    sin_solar_ze = np.sin(solar_ze_rad)
    x_solar = sin_solar_ze * np.sin(solar_az_rad)
    y_solar = sin_solar_ze * np.cos(solar_az_rad)
    z_solar = np.cos(solar_ze_rad)
    return np.stack((x_solar, y_solar, z_solar), axis=0)


class SingleaxisTrackerWSlope():
    """
    SingleAxis Tracekr with Slope

    Parmeters
    ---------
    system_plane : tuple of float
        the orientation of the plane containing the tracker axes, should be a
        tuple of floats with azimuth and zenith 
    """
        
    def __init__(self, system_plane, tracker_azimuth, max_rotation, gcr):
        self.system_azimuth = np.radians(system_plane[0])  #: system azimuth
        self.system_zenith = np.radians(system_plane[1])  #: system zenith
        self.tracker_azimuth = np.radians(tracker_azimuth)  #: tracker aximuth
        self.max_rotation = np.radians(max_rotation)  #: maximum rotation
        self.gcr = gcr  #: gcr
        # z-rotation matrix global to system plane
        self._sys_z_rot = get_rotation_matrix(self.system_azimuth, 'z')
        # x-rotation matrix global to system plane
        self._sys_x_rot = get_rotation_matrix(self.system_zenith, 'x')
        #: tracker axis zenith
        self.tracker_zenith = -self._calc_tracker_axis_tilt()
        # tracker axis rotation relative to system plane
        self._sys_track_rel_rot = self._calc_system_tracker_relative_rotation()
        # z-rotation matrix system plane to tracker
        self._sys_tr_z_rot = get_rotation_matrix(self._sys_track_rel_rot, 'z')
        #: tracker side slope
        self.tracker_side_slope = self._calc_side_slope()

    def _calc_tracker_axis_tilt(self):
        # calculate tracker axis tilt in the global reference frame:
        #
        # 1. the trackers axes are in the system plane, so the z-coord = 0
        #
        # 2. rotate the trackers [x_tr_sys, y_tr_sys, 0] back to the global,
        #    but rotated by the tracker global azimuth if there is one, so that
        #    the tracker axis is constrained to y-z plane so that x-coord = 0
        #
        # Rx_sys = [[1,           0,            0],
        #           [0, cos(sys_ze), -sin(sys_ze)],
        #           [0, sin(sys_ze),  cos(sys_ze)]]
        #
        # Rz_sys = [[cos(sys_az-tr_az), -sin(sys_az-tr_az), 0],
        #           [sin(sys_az-tr_az),  cos(sys_az-tr_az), 0],
        #           [                0,                  0, 1]]
        #
        # tr_rot_glo = Rz_sys.T * (Rx_sys.T * [x_tr_sys, y_tr_sys, 0])
        #
        # tr_rot_glo = [
        # [ x_tr_sys*cos(sys_az-tr_az)+y_tr_sys*sin(sys_az-tr_az)*cos(sys_ze)],
        # [-x_tr_sys*sin(sys_az-tr_az)+y_tr_sys*cos(sys_az-tr_az)*cos(sys_ze)],
        # [                                             -y_tr_sys*sin(sys_ze)]]
        #
        # 3. solve for x_tr_sys
        #
        # x_tr_sys*cos(sys_az-tr_az)+y_tr_sys*sin(sys_az-tr_az)*cos(sys_ze) = 0
        # x_tr_sys = -y_tr_sys*tan(sys_az-tr_az)*cos(sys_ze)
        #
        # 4. so tracker axis tilt, tr_ze = arctan2(tr_rot_glo_z, tr_rot_glo_y)
        #
        # tr_rot_glo_y = y_tr_sys*cos(sys_ze)*(
        #     tan(sys_az-tr_az)*sin(sys_az-tr_az) + cos(sys_az-tr_az))
        #
        # tan(tr_ze) = -y_tr_sys*sin(sys_ze) / tr_rot_glo_y
        #
        # trick: multiply top and bottom by cos(sys_az-tr_az) and remember that
        #        sin^2 + cos^2 = 1 (or just use sympy.simplify)
        # 
        # tan(tr_ze) = -tan(sys_ze)*cos(sys_az-tr_az)  QED
        sys_az_rel_to_tr_az = self.system_azimuth - self.tracker_azimuth
        tan_tr_ze = -np.cos(sys_az_rel_to_tr_az)*np.tan(self.system_zenith)
        return np.arctan(tan_tr_ze)
        
    def _calc_system_tracker_relative_rotation(self):
        # find the relative rotation of the trackers in the system plane
        # 1. tracker axis vector
        cos_tr_ze = np.cos(-self.tracker_zenith)
        tr_ax = np.array([
            [cos_tr_ze*np.sin(self.tracker_azimuth)],
            [cos_tr_ze*np.cos(self.tracker_azimuth)],
            [np.sin(-self.tracker_zenith)]])
        # 2. rotate tracker axis vector from global to system reference frame 
        tr_ax_sys_Rz = np.dot(self._sys_z_rot, tr_ax)
        tr_ax_sys = np.dot(self._sys_x_rot, tr_ax_sys_Rz)
        return np.arctan2(tr_ax_sys[0, 0], tr_ax_sys[1, 0])

    def _calc_side_slope(self):
        # find side slope
        # 1. tracker normal vector
        sin_tr_ze = np.cos(self.tracker_zenith)
        tr_norm = np.array([
            [sin_tr_ze*np.sin(self.tracker_azimuth)],
            [sin_tr_ze*np.cos(self.tracker_azimuth)],
            [np.cos(self.tracker_zenith)]])
        # 2. rotate tracker normal vector from global to system reference frame 
        tr_norm_sys_Rz = np.dot(self._sys_z_rot, tr_norm)
        tr_norm_sys = np.dot(self._sys_x_rot, tr_norm_sys_Rz)
        # 3. side slope is angle between tracker normal and system plane normal
        # np.arccos(tr_norm_sys[2])
        # 4. but we need to know which way the slope is facing, so rotate to
        # tracker use arctan2
        tr_norm_sys_tr = np.dot(self._sys_tr_z_rot, tr_norm_sys)
        return np.arctan2(tr_norm_sys_tr[0, 0], tr_norm_sys_tr[2, 0])

    def get_tracker_rotation(self, solar_position, backtracking=True):
        ze = solar_position['apparent_zenith'].values
        az = solar_position['azimuth'].values
        is_day = ze < DAY
        solar_vector = get_solar_vector(ze, az)
        # rotate solar vector into system plane coordinate system
        sol_sys_Rz = np.dot(self._sys_z_rot, solar_vector)
        sol_sys = np.dot(self._sys_x_rot, sol_sys_Rz)
        # rotate solar vector into tracker coordinate system
        sol_sys_tr = np.dot(self._sys_tr_z_rot, sol_sys)
        # tracker rotation without limits
        tr_rot_no_lim = np.arctan2(sol_sys_tr[0, :], sol_sys_tr[2, :])
        tr_rot_rad = np.maximum(-self.max_rotation, tr_rot_no_lim)
        tr_rot_rad = np.minimum(tr_rot_rad, self.max_rotation)
        if backtracking:
            # this could be a place to try the walrus := operator from py38
            lx = np.cos(tr_rot_rad)
            is_backtrack = lx < self.gcr
            is_backtrack = np.logical_and(is_backtrack, is_day)
            backtrack_rot = np.where(is_backtrack, np.arccos(lx / self.gcr), 0)
            tr_rot_backtrack = tr_rot_rad - backtrack_rot * np.sign(tr_rot_rad)
        else:
            tr_rot_backtrack = tr_rot_rad
        # calculate angle of incidence
        x_tracker = np.sin(tr_rot_backtrack)
        z_tracker = np.cos(tr_rot_backtrack)
        aoi_rad = np.arccos(
            x_tracker*sol_sys_tr[0, :] + z_tracker*sol_sys_tr[2, :])
        tr_rot_horz = self.tracker_side_slope - tr_rot_backtrack
        tracker_rotation = np.degrees(tr_rot_horz)
        aoi = np.degrees(aoi_rad)
        # TODO: output surface normal vector orientation (az, ze)
        return tracker_rotation, aoi, tr_rot_rad


def test_tracker_rotation():
    singleaxis_tracker_wslope_test = SingleaxisTrackerWSlope(
        system_plane=(77.34, 10.1149),
        tracker_azimuth=0,
        max_rotation=75,
        gcr = 0.328
    )
    assert np.isclose(
        singleaxis_tracker_wslope_test.system_azimuth, 1.349837643)
    assert np.isclose(
        singleaxis_tracker_wslope_test.system_zenith, 0.176538309)
    assert np.isclose(singleaxis_tracker_wslope_test.tracker_azimuth, 0.0)
    assert np.isclose(singleaxis_tracker_wslope_test.max_rotation, 1.308996939)
    LOGGER.debug(
        'sideslope = %g', singleaxis_tracker_wslope_test.tracker_side_slope)
    assert np.isclose(
        singleaxis_tracker_wslope_test.tracker_side_slope, -0.172202784)
    LOGGER.debug(
        'tracker zenith = %g', singleaxis_tracker_wslope_test.tracker_zenith)
    assert np.isclose(
        singleaxis_tracker_wslope_test.tracker_zenith, 0.039077922)
    starttime = '2017-01-01T00:30:00-0300'
    stoptime = '2017-12-31T23:59:59-0300'
    lat, lon = -27.597300, -48.549610
    times = pd.DatetimeIndex(pd.date_range(starttime, stoptime, freq='H'))
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    trrot, aoi, trrot_rad = singleaxis_tracker_wslope_test.get_tracker_rotation(solpos)
    expected = pd.read_csv('Florianopolis_Brasilia.csv')
    assert np.allclose(solpos['apparent_zenith'], expected['zen'])
    assert np.allclose(solpos['azimuth'], expected['azim'])
    assert np.allclose(trrot, expected['trrot'].values)
    aoi90 = np.abs(aoi) < 90
    assert np.allclose(aoi[aoi90], expected['aoi'][aoi90].values, 0.00055)
    return trrot, aoi, trrot_rad, singleaxis_tracker_wslope_test


def test_pvlib_flat():
    starttime = '2017-01-01T00:30:00-0300'
    stoptime = '2017-12-31T23:59:59-0300'
    lat, lon = -27.597300, -48.549610
    times = pd.DatetimeIndex(pd.date_range(starttime, stoptime, freq='H'))
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    pvlib_flat = pvlib.tracking.singleaxis(
        solpos['apparent_zenith'], solpos['azimuth'])
    sat_flat = SingleaxisTrackerWSlope((0,0), 0, max_rotation=90, gcr=2.0/7.0)
    trrot, aoi, _ = sat_flat.get_tracker_rotation(solpos)
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
    sat_tilt20 = SingleaxisTrackerWSlope((180.0, 20.0), 0, max_rotation=90, gcr=2.0/7.0)
    trrot, aoi, _ = sat_tilt20.get_tracker_rotation(solpos)
    nans = np.isnan(pvlib_tilt20['tracker_theta'])
    # FIXME: pvlib and sat are not agreeing on some backtracking times
    ninetys = np.abs(pvlib_tilt20['tracker_theta']) < 90.000000
    zeroes = np.isclose(trrot, 0.0)
    conditions = ~nans & ninetys & ~zeroes
    # TODO: now both are pointing south, and signs are the same, look into this
    assert np.allclose(pvlib_tilt20['tracker_theta'][conditions], trrot[conditions])
    assert np.allclose(pvlib_tilt20['aoi'][conditions], aoi[conditions])


if __name__ == "__main__":
    trrot, aoi, trrot_rad, singleaxis_tracker_wslope_test = test_tracker_rotation()
    test_pvlib_flat()
    test_pvlib_tilt20()
