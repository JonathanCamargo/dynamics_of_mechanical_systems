#!/usr/bin/env python
"""
Launch the RRR robot real-time viewer.

Usage:
    python run_gui.py
"""

from dms.robots import rrr
from dms.robots.model import JointInfo, ServoCal
from dms.robots.robot_viewer import launch

if __name__ == "__main__":
    print("RRR Viewer")
    print("=" * 40)    
    
    # Setup for senecabot arm:
    params = {
         # Link lengths (m)
         "LA01": 80e-3,
         "LB01": 80e-3,
         "LC01": 40e-3, 
         # Construction offsets: angle from the modelled frame axis to the
         # physical link centreline, deg, positive CCW.  Measure once from CAD
         # or the real arm.  All zero => theta_A from +x, theta_B/theta_C
         # relative with 0 = collinear with the previous link.
         "alpha_A_deg": 146.0,
         "alpha_B_deg": 180.0,
         "alpha_C_deg": 180.0, 
         # Rotation sense of each joint angle: +1 = CCW positive (matches the
         # viewer and makes omega_C = (thA' + thB' + thC') z), -1 = CW positive,
         # for a joint whose useful travel is one-sided.
         "sign_A": +1,
         "sign_B": -1,
         "sign_C": +1,
     }

    joints = [
            JointInfo("Joint A", 47, 97, 97),
            JointInfo("Joint B", 37, 120, 120),
            JointInfo("Joint C", 31, 141, 141),
        ]

    servos = [
        ServoCal(0,  q_ref=97, servo_ref=90, gain=-1),
        ServoCal(1,  q_ref=120, servo_ref=90,gain=1),
        ServoCal(2, q_ref=141, servo_ref=90,gain=-1),
    ]
    
    robot = rrr(params=params,joints=joints,servos=servos)        
    
    launch(robot, title="RRR Viewer")
