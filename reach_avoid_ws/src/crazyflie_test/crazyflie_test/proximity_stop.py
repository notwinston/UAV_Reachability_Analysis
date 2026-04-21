  #!/usr/bin/env python

from crazyflie_py import Crazyswarm
import numpy as np

TAKEOFF_HEIGHT = 1.0
TAKEOFF_DURATION = 3.0
STOP_DISTANCE = 2.0    # meters — drones stop when this close
APPROACH_SPEED = 0.4   # m/s used to estimate goTo duration
LAND_HEIGHT = 0.04
LAND_DURATION = 3.0


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    assert len(allcfs.crazyflies) == 2, \
        f'Expected 2 drones, got {len(allcfs.crazyflies)}'

    cf1 = allcfs.crazyflies[0]
    cf2 = allcfs.crazyflies[1]

    # --- Takeoff ---
    print('Taking off...')
    allcfs.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 1.5)

    # --- Countdown ---
    for i in range(5, 0, -1):
        print(i)
        timeHelper.sleep(1.0)
    print('0 - GO!')

    # Sample positions after takeoff has settled
    pos1 = np.array(cf1.position)
    pos2 = np.array(cf2.position)
    pos1[2] = TAKEOFF_HEIGHT
    pos2[2] = TAKEOFF_HEIGHT

    separation = np.linalg.norm(pos2 - pos1)
    if separation < 0.01:
        print('ERROR: drones appear to be at the same position — check crazyflies.yaml')
        allcfs.land(targetHeight=LAND_HEIGHT, duration=LAND_DURATION)
        timeHelper.sleep(LAND_DURATION + 1.0)
        return
   
    # Each drone targets the other's starting position — they fly directly at each other
    approach_duration = separation / APPROACH_SPEED

    cf1.goTo(pos2, 0.0, approach_duration)
    cf2.goTo(pos1, 0.0, approach_duration)
    print(f'Drones approaching — separation {separation:.2f} m, '
          f'approach duration {approach_duration:.1f} s')

    # --- Monitor until within STOP_DISTANCE ---
    stopped = False
    while not timeHelper.isShutdown():
        cur1 = np.array(cf1.position)
        cur2 = np.array(cf2.position)
        d = float(np.linalg.norm(cur2 - cur1))
        print(f'  dist={d:.3f} m  CF1={np.round(cur1, 3)}  CF2={np.round(cur2, 3)}')

        if d <= STOP_DISTANCE:
            print(f'Within {STOP_DISTANCE} m — hovering in place.')
            # Override goTo with current position to stop each drone immediately
            cf1.goTo(cur1.tolist(), 0.0, 1.5)
            cf2.goTo(cur2.tolist(), 0.0, 1.5)
            stopped = True
            break

        timeHelper.sleep(0.05)  # ~20 Hz

    if not stopped:
        print('Loop exited without triggering stop (shutdown?)')

    # Hold hover briefly so the position commands settle
    timeHelper.sleep(2.0)

    # --- Land ---
    print('Landing...')
    allcfs.land(targetHeight=LAND_HEIGHT, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION + 1.0)
    print('Done.')


if __name__ == '__main__':
    main()
