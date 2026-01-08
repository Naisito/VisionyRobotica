#!/usr/bin/env python3

import numpy as np 
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
    
    def mover_lineal(self, pose: Pose) -> bool:
        return self.mover_trayectoria([pose])
        
    def subir(self, cantidad: float) -> bool:
        pose_act = self.pose_actual()
        pose_act.position.z += cantidad
        
        return self.mover_lineal(pose_act)
    
    def bajar(self, cantidad: float) -> bool:
        return self.subir(-cantidad)
    

if __name__ == '__main__':
    from poses import torre1, torre2, punto0
    from poses import torre1, torre2
    
    node = NodoRobot()
    
    suelo = node.añadir_suelo()
   
    node.añadir_caja_a_escena_de_planificacion(torre1,"torre1",(.07,.07,.77))

    node.añadir_caja_a_escena_de_planificacion(torre2,"torre2",(.16,.76,.23))

    home_nuestro = [2.103938102722168, -1.9056993923582972, 1.328787628804342, -0.9948828977397461, -1.5678799788104456, 0.39615392684936523]
    node.mover_articulaciones(home_nuestro)
    pose_home = node.pose_actual()
    pose_home.position.x = -0.40412
    pose_home.position.y = 0.1709
    #pose_home.position.z -= .1

    '''
    Prueba1
        "x_mm": -220,07
        "y_mm": 127.1

        x= 0.29918
        y= 0.2485

        esperado:
        x= 0.34918
        y= 0.2285
    
    Prueba2
        "x_mm": -556.45,
        "y_mm": 122.19
        
        x= 0.0372
        y= 0.24359

        esperado:
        x= -0.08
        y= -0.02
        
    Prueba3
        "x_mm": -218.67,
        "y_mm": 236.65

        x= 0.30058
        y= 0.35805
        
        esperado:
        x= -0.04
        y= 0.04
        
    Prueba4
        "x_mm": -115.13,
        "y_mm": 49.5

        x=0.40412
        y=0.1709
        
        no llega sin cambiar z
        
    Prueba5
        
            
            
    '''

    node.mover_articulaciones(pose_home)


    home_nuestro = [2.103938102722168, -1.9056993923582972, 1.328787628804342, -0.9948828977397461, -1.5678799788104456, 0.39615392684936523]
    node.mover_articulaciones(home_nuestro)