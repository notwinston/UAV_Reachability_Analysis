#!/usr/bin/env python3
"""Simple takeoff script for Crazyflie via Crazyswarm2 /hw/takeoff service."""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


def main():
    rclpy.init()
    node = Node('takeoff_client')
    client = node.create_client(Trigger, '/hw/takeoff')

    node.get_logger().info('Waiting for /hw/takeoff service...')
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error('/hw/takeoff service not available. Is crazyswarm_adapter running?')
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info('Sending takeoff request (1.0m)...')
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

    if future.result() is not None:
        resp = future.result()
        if resp.success:
            node.get_logger().info(f'Takeoff successful: {resp.message}')
        else:
            node.get_logger().warn(f'Takeoff rejected: {resp.message}')
    else:
        node.get_logger().error('Takeoff service call timed out')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
