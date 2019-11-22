#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Single Axis Tracker with slope"""

import logging
import numpy as np
import pandas as pd
import pvlib

logging.basicConfig()

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def _get_rotation_matrix(angle, axis=0):
    """
    Return a rotation matrix that when multiplied by a column vector returns
    a new column vector that is rotated clockwise around the given axis by the
    given angle.

    Parameters
    ----------
    angle : float
        Angle of rotation [radians]
    axis : int, default 0
        Axis of rotation, 0=x, 1=y, 2=z

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
    sin_tr_az = np.sin(tracker_azimuth)
    cos_tr_az = np.cos(tracker_azimuth)
    tr_ax = np.array([
        [cos_tr_ze*sin_tr_az],
        [cos_tr_ze*cos_tr_az],
        [np.sin(-tracker_zenith)]])
    # 2. rotate tracker axis vector from global to system reference frame
    sys_z_rot = _get_rotation_matrix(system_azimuth, axis=2)
    # first around the z-axis
    tr_ax_sys_z_rot = np.dot(sys_z_rot, tr_ax)
    # then around x-axis so that xy-plane is the plane with slope and trackers
    sys_x_rot = _get_rotation_matrix(system_zenith)
    tr_ax_sys = np.dot(sys_x_rot, tr_ax_sys_z_rot)
    # now that tracker axis is in coordinate system of slope, the relative
    # rotation is the angle from the y axis
    tr_rel_rot = np.arctan2(tr_ax_sys[0, 0], tr_ax_sys[1, 0])
    # find side slope
    # 1. tracker normal vector
    sin_tr_ze = np.sin(tracker_zenith)
    tr_norm = np.array([
        [sin_tr_ze*sin_tr_az],
        [sin_tr_ze*cos_tr_az],
        [cos_tr_ze]])  # note: cos(-x) = cos(x)
    # 2. rotate tracker normal vector from global to system reference frame
    tr_norm_sys_z_rot = np.dot(sys_z_rot, tr_norm)
    tr_norm_sys = np.dot(sys_x_rot, tr_norm_sys_z_rot)
    # 3. side slope is angle between tracker normal and system plane normal
    # np.arccos(tr_norm_sys[2])
    # 4. but we need to know which way the slope is facing, so rotate to
    # tracker use arctan2
    sys_tr_z_rot = _get_rotation_matrix(tr_rel_rot, axis=2)
    tr_norm_sys_tr = np.dot(sys_tr_z_rot, tr_norm_sys)
    side_slope = np.arctan2(tr_norm_sys_tr[0, 0], tr_norm_sys_tr[2, 0])
    return side_slope, tr_rel_rot
