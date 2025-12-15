#!/usr/bin/env python3

import time
import sys
import copy
import rospy
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, PlanningSceneInterface
import moveit_msgs.msg
from math import pi, tau, dist, fabs, cos
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from typing import List
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from tf.transformations import quaternion_from_euler
from control_msgs.msg import GripperCommandAction, GripperCommandGoal, GripperCommandResult
from actionlib import SimpleActionClient

class NodoRobot:
    def __init__(self) -> None:
        roscpp_initialize(sys.argv)
        rospy.init_node("control_robot", anonymous=True)
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group_name = "robot"
        self.move_group = MoveGroupCommander(self.group_name)
        self.gripper_action_client = SimpleActionClient("rg2_action_server", GripperCommandAction)
        self.añadir_suelo()

    def articulaciones_actuales(self) -> list:
        return self.move_group.get_current_joint_values()
    
    def mover_articulaciones(self, joint_goal: List[float], wait: bool= True) -> bool:
        return self.move_group.go(joint_goal, wait=wait)
    
    def pose_actual(self) -> Pose:
        return self.move_group.get_current_pose().pose
    
    def pose_a_stamped(self, pose: Pose) -> PoseStamped:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.pose = pose
        return pose_stamped
    
    def mover_a_pose(self, pose_goal: Pose, wait: bool=True) -> bool:
        self.move_group.set_pose_target(pose_goal)
        return self.move_group.go(wait=wait)
    
    def añadir_caja_a_escena_de_planificacion(self, pose_caja: Pose, name: str,
                                  tamaño: tuple = (.1,.1,.1)) -> None:
        box_pose = PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose = pose_caja
        box_name = name
        self.scene.add_box(box_name, box_pose, size=tamaño)

    def mover_trayectoria(self, poses: List[Pose], wait: bool = True) -> bool:
        poses_aux = copy.deepcopy(poses)
        poses_aux.insert(0, self.pose_actual())
            
        (plan, fraction) = self.move_group.compute_cartesian_path(poses_aux, 0.01)

        if fraction != 1.0:
            return False
        
        return self.move_group.execute(plan, wait=wait)

    def añadir_suelo(self) -> None:
        pose_suelo = Pose()
        pose_suelo.position.z = -0.026
        self.añadir_caja_a_escena_de_planificacion(pose_suelo,"suelo",(2,2,.05))
        
    def mover_pinza(self, anchura_dedos: float, fuerza: float) -> bool:
        goal = GripperCommandGoal()
        goal.command.position = anchura_dedos
        goal.command.max_effort = fuerza
        self.gripper_action_client.send_goal(goal)
        self.gripper_action_client.wait_for_result()
        result = self.gripper_action_client.get_result()
        
        return result.reached_goal


if __name__ == '__main__':
    node = NodoRobot()
    # home = [2.384185791015625e-07, -1.57081999401235, 1.1269246236622621e-05, -1.5708099804320277, 1.7702579498291016e-05, 3.320696851005778e-05]
    # node.mover_articulaciones(home)
    # time.sleep(5)
    # art1 = [2.103938102722168, -1.9056993923582972, 1.328787628804342, -0.9948828977397461, -1.5678799788104456, 0.39615392684936523]
    # node.mover_articulaciones(art1)
    suelo = node.añadir_suelo()
    """
    basura = node.pose_actual()  
    
    # basura.position.x = 
    # basura.position.y = 
    # basura.position.z -= 
    
    # pinza en posición neutra
    basura.orientation.x = 0
    basura.orientation.y = 0
    basura.orientation.z = 0
    basura.orientation.w = 1
    print(basura)
    
    x,y,z,w = quaternion_from_euler(ai=90, aj=0, ak=0) # para hacer rotaciones raras
    """
    basura = Pose(position= Point(x=-0.01652226419654452,
                                  y =0.33709295103918974,
                                  z= .05), # .3
                  orientation = Quaternion(x = 0,
                                           y= 0,
                                           z=0,
                                           w=1))
    # 32/24/12
    node.añadir_caja_a_escena_de_planificacion(basura,"basura",(.24,.36,.12))
   
    torre1 = Pose(position= Point(x=-0.281471017258501,
                                  y= -0.16223313574084047,
                                  z= 0.35), # 0.6423561887074957
                  orientation = Quaternion(x = 0,
                                           y= 0,
                                           z=0,
                                           w=1))
    node.añadir_caja_a_escena_de_planificacion(torre1,"torre1",(.07,.07,.77))

    torre2 = Pose(position= Point(x=-0.28, # -0.39414925440106247
                                  y= 0.18, # -0.21688130543806292
                                  z= 0.75), # 0.6423561887074957
                  orientation = Quaternion(x = 0,
                                           y= 0,
                                           z=0,
                                           w=1))
    node.añadir_caja_a_escena_de_planificacion(torre2,"torre2",(.16,.76,.23))