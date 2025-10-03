"""
@file test_gimbal_rotation.py
@Description: This is a test script for using the SIYI SDK Python implementation to set/get gimbal rotation
@Author: Mohamed Abdelkader
@Contact: mohamedashraf123@gmail.com
All rights reserved 2022
"""
from operator import truediv
from time import sleep
import sys
import os
  
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
  
sys.path.append(parent_directory)

from siyi_sdk import SIYISDK
from siyi_control import SIYIControl
import threading




def test():

    siyi_control = SIYIControl()

    siyi_control.cam.setYawAngle(40)
    siyi_control.cam.setPitchAngle(90)

    msg = siyi_control.cam.getAttitude()
    sleep(1)

    print("ZOOM: ", msg)

    #siyi_control.cam.setYawAngle(40)

if __name__ == "__main__":
   while True:
       test()