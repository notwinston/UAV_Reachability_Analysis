#!/usr/bin/env python

import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = "radio://0/80/2M/E7E7E7E7E7"

cflib.crtp.init_drivers()

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    cf = scf.cf

    print("Connected")

    # unlock
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)

    start = time.time()

    while time.time() - start < 10:
        vx = 0.0
        vy = 0.0
        vz = 0.8   # go up slowly
        yawrate = 0.0

        cf.commander.send_velocity_world_setpoint(vx, vy, vz, yawrate)
        time.sleep(0.02)  # ~50 Hz

    # stop
    cf.commander.send_stop_setpoint()