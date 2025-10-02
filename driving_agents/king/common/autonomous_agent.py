#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla

from leaderboard.utils.route_manipulation import downsample_route

import torch
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'

import torch
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutonomousAgent(object):
    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, args, device=None, path_to_conf_file=None):
        if device is None:
            device = DEFAULT_DEVICE
        elif isinstance(device, str):
            device = torch.device(device)
        # Fallback if CUDA was requested but not available
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        self.device = device
        self.track = Track.SENSORS
        self._global_plan = None
        self._global_plan_world_coord = None

        # Call setup with consistent args
        self.setup(args=args, device=self.device, path_to_conf_file=path_to_conf_file)

        self.wallclock_t0 = None

    def setup(self, args=None, device=None, path_to_conf_file=None):
        """
        Override in subclasses
        """
        self.device = device or DEFAULT_DEVICE



    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def set_global_plan(self, global_plan_gps_list, global_plan_world_coord_list):
        """
        Set the plan (route) for the agent;  batched

        Parameters:
        - global_plan_gps_list: List of global_plan_gps
        - global_plan_world_coord_list: List of global_plan_world_coord
        """
        self._global_plan_world_coord_list = []
        self._global_plan_list = []

        for global_plan_gps, global_plan_world_coord in zip(global_plan_gps_list, global_plan_world_coord_list):
            ds_ids = downsample_route(global_plan_world_coord, 50)
            self._global_plan_world_coord_list.append([(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids])
            self._global_plan_list.append([global_plan_gps[x] for x in ds_ids])
