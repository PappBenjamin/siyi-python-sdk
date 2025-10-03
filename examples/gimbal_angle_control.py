import sys
import os
import threading
from time import sleep

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from siyi_control import SIYIControl

stop_event = threading.Event()

def data_loop(siyi_control):
    while not stop_event.is_set():
        msg = siyi_control.cam.getAttitude()
        print("Gimbal data:", msg)
        sleep(2)  # Adjust for how often you want to get data

if __name__ == "__main__":
    siyi_control = SIYIControl()
    data_thread = threading.Thread(target=data_loop, args=(siyi_control,))
    data_thread.start()
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        print("Exiting on user interrupt.")
        stop_event.set()
        data_thread.join()
