# Robótica y Automatización Inteligente

Entorno docente basado en ROS empaquetado en Docker para la asignatura de Robótica y Automatización Inteligente (Máster de Computación y Sistemas Inteligentes, Universidad de Deusto). Incluye un demo completo de clasificación de residuos con visión y manipulación robótica.

## Prerrequisitos mínimos
- Docker instalado.
- VS Code + extensión Remote - Containers.
- Git.
- SO: Linux/macOS/Windows con Docker; para GPU: Linux/WSL con X11 + NVIDIA Container Toolkit.
- Puertos: 6081 abierto para noVNC (perfil desktop). Contraseña noVNC: laboratorio.

## Qué hay en este repositorio
- Espacio de trabajo ROS listo para construir (`ros_workspace`).
- Imágenes Docker con perfiles de ejecución (desktop, local, local_gpu).
- Demo de clasificación de residuos controlando un UR3e y la pinza RG2.

## Estructura rápida
- [ros_workspace](ros_workspace): workspace principal con paquetes, lanzadores y recursos.
- [ros_workspace/src/launcher_robots_lab_robotica](ros_workspace/src/launcher_robots_lab_robotica): lanzadores, URDF y configuraciones de control de los UR3e.
- [ros_workspace/src/ros_python_pkg-main](ros_workspace/src/ros_python_pkg-main): nodos Python, requisitos y utilidades.
- [ros_workspace/src/universal_robot](ros_workspace/src/universal_robot) y [ros_workspace/src/Universal_Robots_ROS_Driver](ros_workspace/src/Universal_Robots_ROS_Driver): descripciones, drivers y MoveIt! para los UR.
- [ejemplos](ejemplos): scripts de ejemplo para publicar, suscribir y controlar robot/cámara desde el host.

## Perfiles de contenedor
| Perfil       | Uso recomendado                               | Requisitos SO/HW                                    | Acceso gráfico                         |
|--------------|-----------------------------------------------|------------------------------------------------------|----------------------------------------|
| desktop      | Uso personal o fuera del laboratorio          | Cualquier SO con Docker (sin X11 ni GPU obligatoria) | noVNC http://localhost:6081 + VS Code  |
| local        | PCs del laboratorio sin GPU                   | Linux/WSL con X11                                   | VS Code Remote; salida gráfica vía X11 |
| local_gpu    | PCs del laboratorio con GPU NVIDIA            | Linux/WSL con X11 + nvidia-container-toolkit         | VS Code Remote; salida gráfica vía X11 |

## Proyecto: sistema de clasificación de residuos
Sistema que detecta residuos (latas, cartones, botellas), estima sus coordenadas mediante dos cámaras y ordena a un UR3e que los deposite en el contenedor correcto.

### Arquitectura (topics)
1. **`nodoCentral`**
   - Suscribe: `gesto` (std_msgs/String).
   - Publica: `nc_gestos` (std_msgs/String).
2. **`nodoObjetos`**
   - Suscribe: `nc_gestos`, `/cam1/usb_cam1/image_raw` (cenital XY), `/cam2/usb_cam1/image_raw` (horizontal Z).
   - Publica: `coordenadas_objetos` (std_msgs/Float32MultiArray) con ID, X, Y, altura.
3. **`nodoRobot`**
   - Suscribe: `nc_gestos`, `coordenadas_objetos`.
   - Controla: UR3e vía MoveIt! y pinza RG2.

### Flujo completo
1) Llega un gesto en `gesto` ("lata" | "carton" | "botella"); `nodoCentral` lo reenvía a `nc_gestos`.
2) `nodoObjetos` captura, clasifica, calcula puntos de agarre, mide con ArUco y publica `coordenadas_objetos`.
3) `nodoRobot` toma la coordenada, se aproxima, agarra, transporta al destino del residuo, deposita y vuelve a HOME.

## Puesta en marcha rápida (recomendada: perfil `desktop`)
### 1) Herramientas
- Docker, VS Code, extensión Remote - Containers y Git instalados.
- Opcional GPU (Linux/X11): NVIDIA Container Toolkit.

### 2) Clonar el repositorio
```bash
git clone https://github.com/Naisito/VisionyRobotica.git
cd VisionyRobotica
```

### 3) Construir la imagen
Elige perfil: `desktop` (cualquier SO), `local_gpu` (laboratorio con NVIDIA), `local` (laboratorio sin GPU).
```bash
docker compose --profile <perfil> build
```

### 4) Lanzar el contenedor
```bash
docker compose --profile <perfil> up
```
- Detén con `Ctrl+C` o gestiona desde la extensión Remote Explorer.
<p align="center">
    <img src="pictures/encender_contenedor.gif" alt="lanzar">
</p>

### 5) Acceder al entorno
- Perfil `desktop`: navegador en http://localhost:6081 (contraseña: laboratorio) para el escritorio noVNC.
- Todos los perfiles: adjunta VS Code al contenedor con Remote Explorer para desarrollar dentro.
<p align="center">
    <img src="pictures/conectarse_contenedor_vscode.gif" alt="Conectarse">
</p>

### 6) Preparar el workspace ROS
Dentro del contenedor (terminal de VS Code):
```bash
cd /home/laboratorio/ros_workspace
sudo apt update
rosdep update --include-eol-distros
rosdep install --from-paths src --ignore-src -r -y
catkin build
source /home/laboratorio/ros_workspace/devel/setup.bash
```
> Activa el workspace en cada terminal (`source .../setup.bash`) o añádelo a `/home/laboratorio/.bashrc`.

### 7) Validar la instalación
```bash
roslaunch launcher_robots_lab_robotica sim_203.launch
```
Deberías ver el UR3e en MoveIt!, planificar con el grupo `robot` y ejecutar trayectorias.

### Notas al trabajar con contenedores
- Evita `docker compose down`: elimina el contenedor y todo lo instalado dentro.
- Lo modificado en `/home/laboratorio/ros_workspace` se comparte con el host.
- Mantén un único contenedor por perfil para no duplicar entornos.

## Referencias
- API Python de MoveIt!: https://moveit.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html
