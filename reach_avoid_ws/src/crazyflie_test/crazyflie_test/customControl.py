# #!/usr/bin/env python

# import time
# import cflib.crtp
# from cflib.crazyflie import Crazyflie
# from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# URI = "radio://0/80/2M/E7E7E7E7E7"

# cflib.crtp.init_drivers()

# with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
#     cf = scf.cf

#     print("Connected")

#     # unlock
#     cf.commander.send_setpoint(0, 0, 0, 0)
#     time.sleep(0.1)

#     start = time.time()

#     while time.time() - start < 10:
#         vx = 0.0
#         vy = 0.0
#         vz = 0.8   # go up slowly
#         yawrate = 0.0

#         cf.commander.send_velocity_world_setpoint(vx, vy, vz, yawrate)
#         time.sleep(0.02)  # ~50 Hz

#     # stop
#     cf.commander.send_stop_setpoint()

from crazyflie_py import Crazyswarm
import numpy as np

Z = 1.0

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    for cf in allcfs.crazyflies:
        cf.setParam("kalman.resetEstimation", 1)
    timeHelper.sleep(0.1)
    for cf in allcfs.crazyflies:
        cf.setParam("kalman.resetEstimation", 0)

    print("Waiting for estimator to stabilize — verify positions look stable:")
    start = timeHelper.time()
    while timeHelper.time() - start < 3.0:
        for cf in allcfs.crazyflies:
            print(f"  [pre-flight] {cf.prefix} pos={cf.position}")
        timeHelper.sleep(0.5)

    print(f"Taking off to {Z:.1f} m...")
    for cf in allcfs.crazyflies:
        cf.takeoff(targetHeight=Z, duration=Z+1.0)

    start = timeHelper.time()
    while timeHelper.time() - start < Z+2.0:
        for cf in allcfs.crazyflies:
            print(f"  [takeoff] {cf.prefix} alt={cf.position[2]:.3f}")
        timeHelper.sleep(0.2)

    for cf in allcfs.crazyflies:
        hover_pos = np.array(cf.initialPosition) + np.array([0, 0, Z])
        cf.goTo(hover_pos, 0, 1.0)
    timeHelper.sleep(1.5)

    print("Hovering...")
    start = timeHelper.time()
    while timeHelper.time() - start < 5.0:
        for cf in allcfs.crazyflies:
            print(f"  [hover]   {cf.prefix} alt={cf.position[2]:.3f}")
        timeHelper.sleep(0.2)

    print("Landing...")
    for cf in allcfs.crazyflies:
        cf.land(targetHeight=0.04, duration=Z+1.0)

    start = timeHelper.time()
    while timeHelper.time() - start < Z+2.0:
        for cf in allcfs.crazyflies:
            print(f"  [land]    {cf.prefix} alt={cf.position[2]:.3f}")
        timeHelper.sleep(0.2)

    print("Done.")

if __name__ == "__main__":
    main()