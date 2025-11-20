#!/usr/bin/env python3
"""central_listener.py

Nodo base que se suscribe dinámicamente a todos los tópicos disponibles
y mantiene el último mensaje recibido por tópico.

Uso:
  - Ejecutar con ROS (noetic/ros1): `rosrun ros_python_pkg central_listener.py` o mediante launch
  - Parametros ROS opcionales:
      ~topics (list): lista de tópicos a suscribir (si no se proporciona, se suscribe a todos)

Este nodo es útil para depuración y para centralizar la recepción de mensajes
cuando múltiples nodos publican en el sistema.
"""

import rospy
from threading import Lock
import rosgraph
import rostopic


class CentralListener(object):
    """Central listener subscribes dynamically to topics and stores last messages."""

    def __init__(self):
        rospy.init_node('central_listener', anonymous=False)

        # Mutex para proteger el acceso a last_messages
        self._lock = Lock()
        # Diccionario topic -> last message
        self.last_messages = {}

        # Obtener parámetro opcional 'topics' (lista de strings)
        topics_param = rospy.get_param('~topics', None)
        if topics_param is not None:
            # Si el usuario pasa un string de tópico único, convertir a lista
            if isinstance(topics_param, str):
                topics_param = [topics_param]
            rospy.loginfo('CentralListener: suscribiéndose a tópicos desde parámetro: %s', topics_param)
            self.topics_to_subscribe = topics_param
        else:
            self.topics_to_subscribe = None

        # Lanzar suscripciones iniciales
        self._setup_subscriptions()

        # Timer para refrescar la lista de tópicos disponibles (por si aparecen nuevos)
        refresh_period = rospy.get_param('~refresh_period', 5.0)
        rospy.Timer(rospy.Duration(refresh_period), self._refresh_subscriptions)

    def _get_system_topics(self):
        """Return list of (topic, topic_type) currently available in the ROS master."""
        try:
            # rostopic.get_topic_list() returns [(topicname, topic_type, num_pub)]
            topics = rostopic.get_topic_list()[0]
            # Normalize to (name, type)
            return [(t[0], t[1]) for t in topics]
        except Exception as e:
            rospy.logwarn('CentralListener: error obteniendo tópicos: %s', e)
            return []

    def _setup_subscriptions(self):
        """Create subscribers for desired topics."""
        system_topics = self._get_system_topics()

        for name, ttype in system_topics:
            # If topics_to_subscribe is specified, skip others
            if self.topics_to_subscribe is not None and name not in self.topics_to_subscribe:
                continue

            # Skip rosout topic to reduce noise
            if name == '/rosout' or name.startswith('/rosout_'):
                continue

            # If we already have a subscriber, skip
            if name in self.last_messages:
                continue

            # Try to create a subscriber with automatic message type resolution
            try:
                msg_class, real_topic, _ = rostopic.get_topic_type(name)
                if msg_class is None:
                    rospy.logdebug('CentralListener: no se pudo resolver tipo para %s', name)
                    continue

                # Create subscriber; using lambda + default arg to capture topic name
                rospy.Subscriber(name, msg_class, lambda msg, tn=name: self._callback(tn, msg))
                rospy.loginfo('CentralListener: suscrito a %s (%s)', name, msg_class.__name__)
                # Initialize last message entry
                with self._lock:
                    self.last_messages[name] = None
            except Exception as e:
                rospy.logwarn('CentralListener: fallo suscripción %s: %s', name, e)

    def _refresh_subscriptions(self, event=None):
        """Periodically refresh subscriptions to catch newly created topics."""
        self._setup_subscriptions()

    def _callback(self, topic_name, msg):
        """Store last message under a lock."""
        with self._lock:
            self.last_messages[topic_name] = msg

    def spin(self):
        """Main loop: mantener el nodo vivo y exponer un servicio simple por consola."""
        rospy.loginfo('CentralListener: listo. use rosparam get /central_listener/last_messages para inspeccionar o echo del topic')
        rospy.spin()


def main():
    node = CentralListener()
    node.spin()


if __name__ == '__main__':
    main()
