#!/usr/bin/env python

"""Test script to check if RobotKinematics can be imported"""

try:
    from lerobot.model.kinematics import RobotKinematics
    print("Import successful!")
    print(f"RobotKinematics class: {RobotKinematics}")
except ImportError as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()