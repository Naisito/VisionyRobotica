#!/usr/bin/env python3

import numpy as np 
import time
import sys
import copy
import rospy
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, PlanningSceneInterface
import moveit_msgs.msg
from math import pi, tau, dist, fabs, cos
from std_msgs.msg import String, Float32MultiArray
from moveit_commander.conversions import pose_to_list
from typing import List
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from tf.transformations import quaternion_from_euler
from control_msgs.msg import GripperCommandAction, GripperCommandGoal, GripperCommandResult
from actionlib import SimpleActionClient

# Coordenadas de referencia del ArUco en el frame del robot (metros)
ARUCO_REF_X = -0.5192520489268268
ARUCO_REF_Y = 0.12140867764445455
Z_MESA = 0.40593990077217484  # Altura Z de la mesa/superficie de trabajo


def mm_to_pose(x_mm: float, y_mm: float, angulo_grados: float, altura_mm: float = 0.0) -> Pose:
    """
    Convierte coordenadas en mm (relativas al ArUco) a una Pose del robot.
    
    La cámara ve las coordenadas relativas al centro del ArUco.
    El robot tiene el ArUco en (ARUCO_REF_X, ARUCO_REF_Y).
    
    Args:
        x_mm, y_mm: Coordenadas en mm relativas al ArUco
        angulo_grados: Ángulo de agarre en grados
        altura_mm: Altura del objeto en mm (para ajustar Z de agarre)
    
    Fórmula:
        robot_x = ARUCO_REF_X - (x_mm / 1000)
        robot_y = ARUCO_REF_Y + (y_mm / 1000)
        robot_z = Z_MESA + altura_mm/1000 (para agarrar por encima del objeto)
    """
    # Convertir mm a metros
    x_m = x_mm / 1000.0
    y_m = y_mm / 1000.0
    altura_m = altura_mm / 1000.0
    
    # Calcular posición en frame del robot
    robot_x = ARUCO_REF_X - x_m
    robot_y = ARUCO_REF_Y + y_m
    robot_z = Z_MESA + altura_m  # Z ajustado con la altura del objeto
    
    pose = Pose()
    pose.position = Point(x=robot_x, y=robot_y, z=robot_z)
    pose.orientation = Quaternion(x=-0.9978007158356853, y=-0.06628522535825553, z=1.6191291997066762e-05, w=1.0716411945185912e-05)
    
    return pose


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
        
        # Lista de poses de objetos detectados
        self.poses_objetos = []
        
        # Gesto recibido (tipo de residuo)
        self.gesto_recibido = None
        
        # Suscriptor al topic de coordenadas
        rospy.Subscriber('coordenadas_objetos', Float32MultiArray, self._coordenadas_cb, queue_size=10)
        rospy.Subscriber('nc_gestos', String, self._gesto_cb, queue_size=10)
        rospy.loginfo("NodoRobot: Suscrito a 'coordenadas_objetos' y 'nc_gestos'")
        
    def _gesto_cb(self, msg: String) -> None:
        """Callback que recibe el gesto (tipo de residuo: lata, carton, botella)."""
        gesto = msg.data.lower().strip()
        self.gesto_recibido = gesto
        rospy.loginfo(f"NodoRobot: Gesto recibido -> '{gesto}'")
    
    def obtener_gesto(self) -> str:
        """Devuelve el último gesto recibido."""
        return self.gesto_recibido

    def _coordenadas_cb(self, msg: Float32MultiArray) -> None:
        """
        Callback que recibe coordenadas de objetos y las convierte a poses.
        
        Formato del array: [id1, x1, y1, angulo1, altura1, id2, x2, y2, angulo2, altura2, ...]
        Cada objeto ocupa 5 posiciones.
        """
        self.poses_objetos = []
        data = msg.data
        
        # Cada objeto tiene 5 valores: id, x, y, angulo, altura
        num_objetos = len(data) // 5
        
        for i in range(num_objetos):
            idx = i * 5
            obj_id = int(data[idx])
            x_mm = data[idx + 1]
            y_mm = data[idx + 2]
            angulo = data[idx + 3]
            altura_mm = data[idx + 4]
            
            # Ahora mm_to_pose usa la altura para calcular Z
            pose = mm_to_pose(x_mm, y_mm, angulo, altura_mm)
            
            self.poses_objetos.append({
                'id': obj_id,
                'x_mm': x_mm,
                'y_mm': y_mm,
                'angulo': angulo,
                'altura_mm': altura_mm,
                'pose': pose
            })
            
            rospy.loginfo(f"NodoRobot: Objeto {obj_id} -> "
                         f"x={pose.position.x:.4f}m, y={pose.position.y:.4f}m, z={pose.position.z:.4f}m, "
                         f"altura={altura_mm:.1f}mm")
        
        rospy.loginfo(f"NodoRobot: Recibidas {num_objetos} poses de objetos")

    def obtener_poses_objetos(self) -> List[dict]:
        """Devuelve la lista de objetos con sus poses."""
        return self.poses_objetos

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
        for i in range(10):
            if (res := self.move_group.go(wait=wait)):
                break
        return res
    
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
    from poses import torre1, torre2, home, basura_carton, basura_botella, basura_lata

    node = NodoRobot()
    # node.subir(0.1)
    
    # node.mover_articulaciones(basura_lata)
    # 1. Añadir suelo (ya se hace en __init__, pero por si acaso)
    rospy.loginfo("Configurando escena de planificación...")
    
    # 2. Añadir las 2 torres
    node.añadir_caja_a_escena_de_planificacion(torre1, "torre1", (.07, .07, .77))
    node.añadir_caja_a_escena_de_planificacion(torre2, "torre2", (.16, .76, .23))
    rospy.loginfo("Torres añadidas a la escena")
    
    # 3. Mover a home
    rospy.loginfo("Moviendo a posición HOME...")
    node.mover_articulaciones(home)
    rospy.loginfo("En posición HOME")
    
    # 4. Abrir garra a 100
    rospy.loginfo("Abriendo garra...")
    node.mover_pinza(100, 20)
    rospy.loginfo("Garra abierta")
    
    # 5. Esperar coordenadas del topic y moverse a ellas
    rospy.loginfo("Esperando coordenadas de objetos en topic 'coordenadas_objetos'...")
    
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        poses = node.obtener_poses_objetos()
        
        if poses:
            rospy.loginfo(f"Recibidas {len(poses)} coordenadas de objetos")
            
            for obj in poses:
                obj_id = obj['id']
                pose = obj['pose']
                pose.position.x -= 0.02
                
                
                rospy.loginfo(f"Moviendo al objeto {obj_id}: "
                             f"x={pose.position.x:.4f}m, y={pose.position.y:.4f}m")
                
                # 6. Mover al objeto
                success = node.mover_a_pose(pose)
                
                if success:
                    rospy.loginfo(f"Llegué al objeto {obj_id}")
                    pose_inicial = node.pose_actual()
                    
                    rospy.loginfo("Moviendo a basura...")
                    tipo_residuo = node.obtener_gesto()
                    if (tipo_residuo == "carton"):
                        pose_inicial.position.z -= 0.1305
                    elif (tipo_residuo == "botella"):
                        pose_inicial.position.z -= 0.17
                    elif (tipo_residuo == "lata"):
                        pose_inicial.position.z -= 0.17
                    else:
                        print('Tipo de residuo desconocido')
                    
                    
                    
                    node.mover_trayectoria([pose_inicial])
                    # 7. Cerrar garra para agarrar el objeto
                    rospy.loginfo("Cerrando garra...")
                    node.mover_pinza(0, 40)
                    
                    rospy.loginfo("Garra cerrada - objeto agarrado")
                    
                    rospy.loginfo("Subiendo")
                    node.subir(0.1)
                    
                    # 8. Mover a posición basura_carton
                    #rospy.loginfo("Moviendo a basura...")
                    #tipo_residuo = node.obtener_gesto()
                    if (tipo_residuo == "carton"):
                        node.mover_articulaciones(basura_carton)
                    elif (tipo_residuo == "botella"):
                        node.mover_articulaciones(basura_botella)
                    elif (tipo_residuo == "lata"):
                        node.mover_articulaciones(basura_lata)
                    else:
                        print('Tipo de residuo desconocido')
                        
                    rospy.loginfo("En posición basura")

                    # 9. Abrir garra para soltar
                    rospy.loginfo("Abriendo garra para soltar...")
                    node.mover_pinza(100, 20)
                    rospy.loginfo("Objeto soltado")
                    
                else:
                    rospy.logwarn(f"No pude llegar al objeto {obj_id}")
            
            # 10. Terminar
            rospy.loginfo("Moviendo a posición HOME...")
            node.mover_articulaciones(home)
            rospy.loginfo("Tarea completada. Finalizando...")
            break  # Salir del bucle y terminar
        
        rate.sleep()
    
    rospy.loginfo("Nodo robot finalizado.")



