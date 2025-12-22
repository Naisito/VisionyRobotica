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
    
    def pose_to_list(self, pose):
        """
        Convierte geometry_msgs/Pose a lista:
        [x, y, z, qx, qy, qz, qw]
        """
        return [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]

    def list_to_pose(self, values):
        """
        Convierte lista [x, y, z, qx, qy, qz, qw] a Pose
        """
        pose = Pose()
        pose.position = Point(values[0], values[1], values[2])
        pose.orientation = Quaternion(
            values[3], values[4], values[5], values[6]
        )
        return pose

if __name__ == '__main__':
    
    node = NodoRobot()    
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
    
    # definir las cosas para que el robot no se choque
    basura = Pose(position= Point(x=-0.01652226419654452,
                                  y =0.38709295103918974,
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
    
    #home = [2.384185791015625e-07, -1.57081999401235, 1.1269246236622621e-05, -1.5708099804320277, 1.7702579498291016e-05, 3.320696851005778e-05]
    #node.mover_articulaciones(home)
    #time.sleep(5)
    
    
    
    # definir donde se tienen que tirar los objetos
   
    basura_carton = Pose(position= Point(x=-0.020229356475544868,
                                  y= 0.3394047330950015,
                                  z= 0.37340989569877925),
                  orientation = Quaternion(x = 0.9999060862391046,
                                           y= -0.013704665897530255,
                                           z=-1.054914307895939e-05,
                                           w=2.68951475182031e-05))
    
    basura_plastico = Pose(position= Point(x=-0.035342447529806524,
                                  y= 0.21990838399222412,
                                  z= 0.43148554119205407),
                  orientation = Quaternion(x = -0.9984570055710325,
                                           y= 0.026806187431695036,
                                           z=-0.0066310189604728694,
                                           w= 0.0481774421181709))
    
    basura_lata = Pose(position= Point(x=-0.055826547870396426,
                                  y= 0.41964205543815697,
                                  z= 0.36653744530851873),
                  orientation = Quaternion(x = -0.9996316787543533,
                                           y= -0.027138658168227518,
                                           z= -7.947575556497002e-06,
                                           w= 6.461414762472053e-07))
    
    prueba = Pose(position= Point(x=-0.44358062856420805,
                                  y= 0.29296848592509883,
                                  z= 0.18524586343853156),
                  orientation = Quaternion(x = -0.9997761137191462,
                                           y= -0.0010203291071483914,
                                           z= 0.009960544306682221,
                                           w= 0.018640518293991038))
    
    prueba2 = Pose(position= Point(x=-0.44358062856420805,
                                  y= 0.29296848592509883,
                                  z= 0.28524586343853156),
                  orientation = Quaternion(x = -0.9997761137191462,
                                           y= -0.0010203291071483914,
                                           z= 0.009960544306682221,
                                           w= 0.018640518293991038))

    home_nuestro = [2.103938102722168, -1.9056993923582972, 1.328787628804342, -0.9948828977397461, -1.5678799788104456, 0.39615392684936523]
    node.mover_articulaciones(home_nuestro)
    time.sleep(10)

    aruco = Pose(position= Point(x=-0.5094436870906449,
                                  y= 0.12080529695346244,
                                  z= 0.2491531422351253),
                  orientation = Quaternion(x = -0.9996350560625232,
                                           y= -0.016719810881794635,
                                           z= 0.020898633025779084,
                                           w= 0.0036673904355839096))
    #node.mover_articulaciones(aruco)
    
    
    #node.mover_articulaciones(prueba)
    #time.sleep(5)
    #node.mover_articulaciones(prueba2)
   
    
    # mover_pinza(mm, fuerza)
    # node.mover_pinza(0.0, 40.0) 
    
    # Mover el robot
    
    pose_inicial = Pose(position= Point(x=-0.377,
                                  y= 0.271,
                                  z= 0.290),
                  orientation = Quaternion(x = 1,
                                           y= -0.03,
                                           z= 0,
                                           w= 0))


    # pose_end = copy.deepcopy(pose_start)
    # pose_end.position.z += 0.03
    
    # Subir el robot
    p1_list = node.pose_to_list(pose_inicial)
    p2_list = p1_list.copy()
    p2_list[2] += 0.05 
    
    trajectory = np.linspace(p1_list, p2_list, 20)
    
    for p in trajectory:
        pose = node.list_to_pose(p)
        node.mover_a_pose(pose)
        
    """
    xs = np.linspace(pose_start.position.x, pose_end.position.x, steps)
    ys = np.linspace(pose_start.position.y, pose_end.position.y, steps)
    zs = np.linspace(pose_start.position.z, pose_end.position.z, steps)
    
    poses = []

    for x, y, z in zip(xs, ys, zs):
        pose = copy.deepcopy(pose_start)
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        poses.append(pose)
    
    node.mover_trayectoria(poses)
    
    print('fin')
    """
    
    
    