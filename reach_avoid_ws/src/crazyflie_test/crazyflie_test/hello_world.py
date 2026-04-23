#!/usr/bin/env python

#!/usr/bin/env python

from pathlib import Path

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import numpy as np


def executeTrajectory(timeHelper, cf, trajpath, rate=100, offset=np.zeros(3)):
    traj = Trajectory()
    traj.loadcsv(trajpath)

    start_time = timeHelper.time()
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        if t > traj.duration:
            break

        e = traj.eval(t)
        cf.cmdFullState(
            e.pos + np.array(cf.initialPosition) + offset,
            e.vel,
            e.acc,
            e.yaw,
            e.omega)

        timeHelper.sleepForRate(rate)


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    rate = 30.0
    Z = 0.5

    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    executeTrajectory(timeHelper, cf,
                      Path(__file__).parent / 'data/figure8.csv',
                      rate,
                      offset=np.array([0, 0, 0.5]))

    cf.notifySetpointsStop()
    cf.land(targetHeight=0.03, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)


if __name__ == '__main__':
    main()


# from pathlib import Path
# import logging
# import time

# from crazyflie_py import Crazyswarm
# from crazyflie_py.uav_trajectory import Trajectory
# import numpy as np


# # ----------------------------
# # Setup Python logging
# # ----------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] [%(levelname)s] %(message)s',
# )
# logger = logging.getLogger(__name__)


# def log_positions(allcfs, prefix=""):
#     for i, cf in enumerate(allcfs.crazyflies):
#         try:
#             pos = cf.position
#             logger.info(f"{prefix} CF{i} pos={np.round(pos, 3)}")
#         except Exception as e:
#             logger.warning(f"{prefix} CF{i} failed to read position: {e}")


# def main():
#     logger.info("=== SCRIPT STARTED ===")

#     swarm = Crazyswarm()
#     timeHelper = swarm.timeHelper
#     allcfs = swarm.allcfs

#     logger.info(f"Number of crazyflies: {len(allcfs.crazyflies)}")

#     traj1 = Trajectory()
#     traj_path = Path(__file__).parent / 'data/figure8.csv'
#     traj1.loadcsv(traj_path)

#     logger.info(f"Loaded trajectory: {traj_path}")
#     logger.info(f"Trajectory duration: {traj1.duration:.2f}s")

#     # ----------------------------
#     # Enable onboard logging (SD card)
#     # ----------------------------
#     logger.info("Enabling onboard USD logging...")
#     allcfs.setParam('usd.logging', 1)

#     TRIALS = 1
#     TIMESCALE = 1.0

#     for i in range(TRIALS):
#         logger.info(f"=== TRIAL {i+1}/{TRIALS} ===")

#         # Upload trajectory
#         logger.info("Uploading trajectory...")
#         for cf in allcfs.crazyflies:
#             cf.uploadTrajectory(0, 0, traj1)

#         # Takeoff
#         logger.info("Taking off...")
#         allcfs.takeoff(targetHeight=1.0, duration=2.0)

#         start = time.time()
#         while time.time() - start < 2.5:
#             log_positions(allcfs, prefix="[TAKEOFF]")
#             timeHelper.sleep(0.2)

#         # Move to start position
#         logger.info("Moving to trajectory start position...")
#         for cf in allcfs.crazyflies:
#             pos = np.array(cf.initialPosition) + np.array([0, 0, 1.0])
#             logger.info(f"GoTo target: {np.round(pos, 3)}")
#             cf.goTo(pos, 0, 2.0)

#         start = time.time()
#         while time.time() - start < 2.5:
#             log_positions(allcfs, prefix="[GOTO]")
#             timeHelper.sleep(0.2)

#         # Start trajectory
#         logger.info("Starting trajectory execution...")
#         allcfs.startTrajectory(0, timescale=TIMESCALE)

#         start = time.time()
#         while time.time() - start < traj1.duration * TIMESCALE + 2.0:
#             log_positions(allcfs, prefix="[TRAJ]")
#             timeHelper.sleep(0.2)

#         # Land
#         logger.info("Landing...")
#         allcfs.land(targetHeight=0.06, duration=2.0)

#         start = time.time()
#         while time.time() - start < 3.0:
#             log_positions(allcfs, prefix="[LAND]")
#             timeHelper.sleep(0.2)

#     # ----------------------------
#     # Disable onboard logging
#     # ----------------------------
#     logger.info("Disabling onboard USD logging...")
#     allcfs.setParam('usd.logging', 0)

#     logger.info("=== SCRIPT DONE ===")


# if __name__ == '__main__':
#     main()

# from crazyflie_py import Crazyswarm

# TAKEOFF_DURATION = 10.0
# HOVER_DURATION = 10.0
# LAND_DURATION = 2.0
# TARGET_HEIGHT = 1.0

# def main():
#     print("SCRIPT STARTED")
#     swarm = Crazyswarm()
#     timeHelper = swarm.timeHelper
#     cf = swarm.allcfs.crazyflies[0]

#     print("Initial pose samples:")
#     for i in range(30):
#         print(f"{i}: pos={cf.position}")
#         timeHelper.sleep(0.1)

#     print("Resetting Kalman filter...")
#     cf.setParam("kalman.resetEstimation", 1)
#     timeHelper.sleep(0.1)
#     cf.setParam("kalman.resetEstimation", 0)
#     timeHelper.sleep(2.0)

#     print("Waiting for estimator to stabilize...")
#     timeHelper.sleep(2.0)

#     print("Post-reset position:", cf.position)

#     print(f"TAKING OFF to {TARGET_HEIGHT:.2f} m")
#     cf.takeoff(targetHeight=TARGET_HEIGHT, duration=TAKEOFF_DURATION)

#     start = timeHelper.time()
#     while timeHelper.time() - start < TAKEOFF_DURATION + HOVER_DURATION:
#         print(f"altitude={cf.position[2]:.3f}, pos={cf.position}")
#         timeHelper.sleep(0.1)

#     print("LANDING")
#     cf.land(targetHeight=0.04, duration=LAND_DURATION)
#     timeHelper.sleep(LAND_DURATION + 1.0)

#     print("DONE")

# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python

# from crazyflie_py import Crazyswarm
# import numpy as np


# def main():

#     for i in range(20):
#         print("position:", cf.position)
#         timeHelper.sleep(0.1)

#     Z = 1.0

#     swarm = Crazyswarm()
#     timeHelper = swarm.timeHelper
#     allcfs = swarm.allcfs

#     allcfs.takeoff(targetHeight=Z, duration=1.0+Z)
#     timeHelper.sleep(1.5+Z)
#     for cf in allcfs.crazyflies:
#         pos = np.array(cf.initialPosition) + np.array([0, 0, Z])
#         cf.goTo(pos, 0, 1.0)

#     print('press button to continue...')
#     swarm.input.waitUntilButtonPressed()

#     allcfs.land(targetHeight=0.02, duration=1.0+Z)
#     timeHelper.sleep(1.0+Z)


# if __name__ == '__main__':
#     main()
