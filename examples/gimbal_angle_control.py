from time import sleep
from examples.siyi_control import SIYIControl_DATA
from pymavlink import mavutil

siyi_control = SIYIControl_DATA()

def get_mavlink_connection():
    """Establish and return a MAVLink connection."""
    try:
        # Use UDP endpoint for QGroundControl forwarding
        mav_conn = mavutil.mavlink_connection('udp:0.0.0.0:14550')
        mav_conn.wait_heartbeat()
        print(f"Heartbeat from system {mav_conn.target_system} component {mav_conn.target_component}")
        return mav_conn
    except Exception as e:
        print(f"Error connecting to MAVLink: {e}")
        return None

def get_gimbal_data():
    """Fetch and print gimbal attitude data."""
    try:
        # siyi_control.cam.requestGimbalInfo()
        # sleep(1)

        attitude = siyi_control.cam.getAttitude()  # returns (yaw, pitch, roll)
        motion = siyi_control.cam.getMotionMode()  # likely returns a value, not a dict
        mountingDir = siyi_control.cam.getMountingDirection()  # likely returns a value, not a dict

        if attitude:
            print(f"Gimbal Attitude: Yaw: {attitude[0]}, Pitch: {attitude[1]}, Roll: {attitude[2]}")
        else:
            print("No attitude data received.")
        if motion is not None:
            print(f"Gimbal Motion Mode: {motion}")
        if mountingDir is not None:
            print(f"Gimbal Mounting Direction: {mountingDir}")

    except Exception as e:
        print(f"Error fetching gimbal attitude: {e}")
        return None

def get_mavlink_telemetry(mav_conn):
    """Fetch and print MAVLink telemetry: battery, attitude, GPS."""
    try:
        msg = mav_conn.recv_match(blocking=True, timeout=5)
        if not msg:
            print("No MAVLink message received.")
            return
        if msg.get_type() == 'BATTERY_STATUS':
            voltage = msg.voltages[0] / 1000.0 if msg.voltages[0] != 0xFFFF else None
            print(f"Battery voltage: {voltage} V")
        elif msg.get_type() == 'ATTITUDE':
            print(f"Attitude - Roll: {msg.roll:.2f}, Pitch: {msg.pitch:.2f}, Yaw: {msg.yaw:.2f}")
        elif msg.get_type() == 'GLOBAL_POSITION_INT':
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.alt / 1000.0
            print(f"GPS - Lat: {lat:.7f}, Lon: {lon:.7f}, Alt: {alt:.2f} m")
    except Exception as e:
        print(f"Error fetching MAVLink telemetry: {e}")



def main():

    print("\n--- SIYI Gimbal Data Fetcher ---")

    # starting services
    # mav_comm = get_mavlink_connection()


    while True:


        # get gixmbal data
        if siyi_control.cam:
            get_gimbal_data()

        # get mavlink telemetry
        # if mav_comm:
        #     get_mavlink_telemetry(mav_comm)


        sleep(1)


if __name__ == "__main__":
    main()
