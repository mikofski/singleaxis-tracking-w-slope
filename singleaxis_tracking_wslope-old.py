#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Single Axis Tracker with slope"""

import logging
import sys
import numpy as np
import pandas as pd
import pvlib

logging.basicConfig()

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
AXES = ['x', 'y', 'z']
DAY = 90.0


def _get_rotation_matrix(angle, axis):
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
       `Rotation Matrix
       <https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations>`_

    """
    r11 = r22 = np.cos(angle)
    r21 = np.sin(angle)
    r12 = -r21
    rot = np.array([
        [1, 0, 0],
        [0, r11, r12],
        [0, r21, r22]])
    axis = AXES.index(axis)
    rot = np.roll(rot, (axis, axis), (1, 0))
    return rot


def _get_solar_vector(solar_zenith, solar_azimuth):
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
        tuple of floats with azimuth and zenith [degrees]
    tracker_azimuth : float
        the direction [degrees] the tracker axes are pointing
    max_rotation : float
        the maximum tracker rotation [degrees] relative to the system plane,
        symmetrical
    gcr : float
        ground coverage ratio, total tracker width perpendicular to axes over
        distance between tracker axes
    """

    def __init__(self, system_plane, tracker_azimuth, max_rotation, gcr):
        self.system_azimuth = np.radians(system_plane[0])  #: system azimuth
        self.system_zenith = np.radians(system_plane[1])  #: system zenith
        self.tracker_azimuth = np.radians(tracker_azimuth)  #: tracker aximuth
        self.max_rotation = np.radians(max_rotation)  #: maximum rotation
        self.gcr = gcr  #: gcr
        # z-rotation matrix global to system plane
        self._sys_z_rot = _get_rotation_matrix(self.system_azimuth, 'z')
        # x-rotation matrix global to system plane
        self._sys_x_rot = _get_rotation_matrix(self.system_zenith, 'x')
        #: tracker axis zenith
        self.tracker_zenith = -self._calc_tracker_axis_tilt()
        # tracker axis rotation relative to system plane
        self._sys_track_rel_rot = self._calc_system_tracker_relative_rotation()
        # z-rotation matrix system plane to tracker
        self._sys_tr_z_rot = _get_rotation_matrix(self._sys_track_rel_rot, 'z')
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
        sin_tr_ze = np.sin(self.tracker_zenith)
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

    def calc_tracker_rotation(self, solar_position, backtracking=True):
        """
        Calculate tracker rotation and angle of incidence in degrees.

        Parameters
        ----------
        solar_position : :class:`pandas.DataFrame`
            the output of ``pvlib.solarposition.get_solarposition``
        backtracking : bool
            if true then backtracking is enabled

        Returns
        -------
        tracker rotation [degrees], angle of incidence [degrees], true-tracking
        [radians]
        """
        ze = solar_position['apparent_zenith'].values
        az = solar_position['azimuth'].values
        is_day = ze < DAY
        solar_vector = _get_solar_vector(ze, az)
        # rotate solar vector into system plane coordinate system
        sol_sys_Rz = np.dot(self._sys_z_rot, solar_vector)
        sol_sys = np.dot(self._sys_x_rot, sol_sys_Rz)
        # rotate solar vector into tracker coordinate system
        sol_sys_tr = np.dot(self._sys_tr_z_rot, sol_sys)
        # tracker rotation without limits
        tr_rot_no_lim = np.arctan2(sol_sys_tr[0, :], sol_sys_tr[2, :])
        if backtracking:
            # this could be a place to try the walrus := operator from py38
            lrot = np.cos(tr_rot_no_lim)  # lrot < 0 if tr_rot > 90[deg]
            is_backtrack = lrot < self.gcr
            is_backtrack = np.logical_and(is_backtrack, is_day)
            cos_rot = np.clip(lrot / self.gcr, -1, 1)  # avoid numpy warnings
            backtrack_rot = np.where(is_backtrack, np.arccos(cos_rot), 0)
            sign_tr_rot = np.sign(tr_rot_no_lim)
            tr_rot_backtrack = tr_rot_no_lim - backtrack_rot * sign_tr_rot
        else:
            tr_rot_backtrack = tr_rot_no_lim
        # appliy rotation limits
        tr_rot_rad = tr_rot_backtrack
        tr_rot_rad = np.maximum(-self.max_rotation, tr_rot_rad)
        tr_rot_rad = np.minimum(tr_rot_rad, self.max_rotation)
        tr_rot_rad[~is_day] = np.nan
        # calculate angle of incidence
        x_tracker = np.sin(tr_rot_rad)
        z_tracker = np.cos(tr_rot_rad)
        aoi_rad = np.where(
            is_day,
            np.arccos(x_tracker*sol_sys_tr[0, :] + z_tracker*sol_sys_tr[2, :]),
            np.nan)
        tr_rot_horz = self.tracker_side_slope - tr_rot_rad
        tracker_rotation = np.degrees(tr_rot_horz)
        aoi = np.degrees(aoi_rad)
        # TODO: output surface normal vector orientation (az, ze)
        return tracker_rotation, aoi, tr_rot_no_lim, tr_rot_backtrack


def test_tracker_rotation():
    singleaxis_tracker_wslope_test = SingleaxisTrackerWSlope(
        system_plane=(77.34, 10.1149),
        tracker_azimuth=0.0,
        max_rotation=75.0,
        gcr=0.328
    )
    assert np.isclose(
        singleaxis_tracker_wslope_test.system_azimuth, 1.349837643)
    assert np.isclose(
        singleaxis_tracker_wslope_test.system_zenith, 0.176538309)
    assert np.isclose(singleaxis_tracker_wslope_test.tracker_azimuth, 0.0)
    assert np.isclose(singleaxis_tracker_wslope_test.max_rotation, 1.308996939)
    LOGGER.debug(
        'sideslope = %g',
        np.degrees(singleaxis_tracker_wslope_test.tracker_side_slope))
    assert np.isclose(
        singleaxis_tracker_wslope_test.tracker_side_slope, -0.172202784)
    LOGGER.debug(
        'tracker zenith = %g',
        np.degrees(singleaxis_tracker_wslope_test.tracker_zenith))
    assert np.isclose(
        singleaxis_tracker_wslope_test.tracker_zenith, 0.039077922)
    starttime = '2017-01-01T00:30:00-0300'
    stoptime = '2017-12-31T23:59:59-0300'
    lat, lon = -27.597300, -48.549610
    times = pd.DatetimeIndex(pd.date_range(starttime, stoptime, freq='H'))
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    trrot, aoi, trrot_nolim, trrot_back \
        = singleaxis_tracker_wslope_test.calc_tracker_rotation(solpos)
    expected = pd.read_csv('Florianopolis_Brasilia.csv')
    assert np.allclose(
        solpos['apparent_zenith'], expected['zen'], equal_nan=True)
    assert np.allclose(
        solpos['azimuth'], expected['azim'], equal_nan=True)
    day = solpos['apparent_zenith'].values < 90
    assert np.allclose(
        trrot[day], expected['trrot'][day].values, equal_nan=True)
    aoi90 = (aoi < 90) & day
    assert np.allclose(
        aoi[aoi90], expected['aoi'][aoi90].values, 0.00055, equal_nan=True)
    return (trrot, aoi, trrot_nolim, trrot_back, solpos,
            singleaxis_tracker_wslope_test)


def test_pvlib_flat():
    starttime = '2017-01-01T00:30:00-0300'
    stoptime = '2017-12-31T23:59:59-0300'
    lat, lon = -27.597300, -48.549610
    times = pd.DatetimeIndex(pd.date_range(starttime, stoptime, freq='H'))
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    pvlib_flat = pvlib.tracking.singleaxis(
        solpos['apparent_zenith'], solpos['azimuth'])
    sat_flat = SingleaxisTrackerWSlope((0, 0), 0, max_rotation=90, gcr=2.0/7.0)
    trrot, aoi, _, _ = sat_flat.calc_tracker_rotation(solpos)
    # FIXME: both pointing north, so why are signs opposite?
    assert np.allclose(pvlib_flat['tracker_theta'], -trrot, equal_nan=True)
    assert np.allclose(pvlib_flat['aoi'], aoi, equal_nan=True)


def test_pvlib_tilt20():
    starttime = '2017-01-01T00:30:00-0300'
    stoptime = '2017-12-31T23:59:59-0300'
    lat, lon = -27.597300, -48.549610
    times = pd.DatetimeIndex(pd.date_range(starttime, stoptime, freq='H'))
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    pvlib_tilt20 = pvlib.tracking.singleaxis(
        solpos['apparent_zenith'], solpos['azimuth'], axis_tilt=20.0,
        axis_azimuth=180.0)
    sat_tilt20 = SingleaxisTrackerWSlope(
        (180.0, 20.0), 0, max_rotation=90, gcr=2.0/7.0)
    trrot, aoi, _, _ = sat_tilt20.calc_tracker_rotation(solpos)
    # FIXME: pvlib and sat are not agreeing on some backtracking times
    ninetys = np.abs(pvlib_tilt20['tracker_theta']) < 90.000000
    zeroes = np.isclose(trrot, 0.0)
    aoi90 = aoi < 90
    conditions = aoi90 & ninetys & ~zeroes
    # TODO: now both are pointing south, and signs are the same, look into this
    assert np.allclose(
        pvlib_tilt20['tracker_theta'][conditions], trrot[conditions],
        equal_nan=True)
    assert np.allclose(
        pvlib_tilt20['aoi'][conditions], aoi[conditions], equal_nan=True)


def test_pvlib_gh656():
    kwargs = dict(apparent_zenith=80, apparent_azimuth=338, axis_tilt=30,
                  axis_azimuth=180, max_angle=60, backtrack=True, gcr=0.35)
    pvlib_gh656 = pvlib.tracking.singleaxis(**kwargs)
    LOGGER.debug('tracker theta = %g', pvlib_gh656['tracker_theta'])
    LOGGER.debug('aoi = %g', pvlib_gh656['aoi'])
    sat_gh656 = SingleaxisTrackerWSlope(
        (kwargs['axis_azimuth'], kwargs['axis_tilt']), kwargs['axis_azimuth'],
        max_rotation=kwargs['max_angle'], gcr=kwargs['gcr'])
    LOGGER.debug('tilt = %g', np.degrees(sat_gh656.tracker_zenith))
    LOGGER.debug('side slope = %g', np.degrees(sat_gh656.tracker_side_slope))
    solpos = pd.DataFrame(
        dict(apparent_zenith=[kwargs['apparent_zenith']],
             azimuth=[kwargs['apparent_azimuth']]),
        index=['2017-01-01T19:00:00-0800'])
    trrot, aoi, _, _ = sat_gh656.calc_tracker_rotation(solpos)
    LOGGER.debug('trrot: %g, aoi: %g', trrot, aoi)
    trrot, aoi, _, _ = sat_gh656.calc_tracker_rotation(
        solpos, backtracking=False)
    LOGGER.debug('trrot: %g, aoi: %g', trrot, aoi)
    sat_gh656 = SingleaxisTrackerWSlope(
        (kwargs['axis_azimuth'], kwargs['axis_tilt']), kwargs['axis_azimuth'],
        max_rotation=180, gcr=kwargs['gcr'])
    trrot, aoi, _, _ = sat_gh656.calc_tracker_rotation(
        solpos, backtracking=True)
    LOGGER.debug('trrot: %g, aoi: %g', trrot, aoi)
    trrot, aoi, _, _ = sat_gh656.calc_tracker_rotation(
        solpos, backtracking=False)
    LOGGER.debug('trrot: %g, aoi: %g', trrot, aoi)


if __name__ == "__main__":
    TRROT, AOI, TRROT_NOLIM, TRROT_BACK, SOLPOS, \
        SINGLEAXIS_TRACKER_WSLOPE_TEST = test_tracker_rotation()
    if len(sys.argv) > 1:
        OUTPUT = {
            'azim': SOLPOS['azimuth'],
            'zen': SOLPOS['apparent_zenith'],
            'trrot': TRROT,
            'aoi': AOI,
            'truetrack': TRROT_NOLIM,
            'backtrack': TRROT_BACK}
        OUTPUT = pd.DataFrame(OUTPUT, index=SOLPOS.index)
        OUTPUT.to_csv(sys.argv[1], index_label='timestamp')
        LOGGER.debug('wrote file: %s', sys.argv[1])
    test_pvlib_flat()
    test_pvlib_tilt20()
    test_pvlib_gh656()
