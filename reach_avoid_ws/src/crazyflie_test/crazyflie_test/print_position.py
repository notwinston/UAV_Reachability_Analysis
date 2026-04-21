from crazyflie_py import Crazyswarm


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper

    print("Printing Vicon position. Press Ctrl+C to stop.")
    while not timeHelper.isShutdown():
        for cf in swarm.allcfs.crazyflies:
            pos = cf.position
            print(f"{cf.prefix}  x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")
        timeHelper.sleepForRate(150)


if __name__ == "__main__":
    main()
