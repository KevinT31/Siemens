# gui.py

import requests
import logging
import threading

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class NodeRedInterface:
    """
    Clase para manejar la comunicación con la interfaz gráfica en Node-RED.
    Permite enviar datos de sensores, recibir comandos de control y actualizar el estado del sistema.
    """

    def __init__(self, node_red_url="http://localhost:1880"):
        """
        Inicializa la interfaz gráfica de Node-RED.
        :param node_red_url: URL base de Node-RED.
        """
        self.node_red_url = node_red_url
        logging.info(f"Interfaz de Node-RED inicializada en {self.node_red_url}.")

    def actualizar_interfaz(self, sensor_values, control_mode):
        """
        Envía los datos de los sensores y el modo de control a Node-RED para su visualización.
        También actualiza el estado de los actuadores y muestra alertas visuales si los valores
        están fuera de los límites.

        :param sensor_values: Diccionario con los valores de los sensores.
        :param control_mode: Modo de control actual ('automatic' o 'manual').
        """
        try:
            data = {
                'sensor_values': sensor_values,
                'control_mode': control_mode
            }
            # Enviar los datos a Node-RED mediante una solicitud POST
            response = requests.post(f"{self.node_red_url}/update", json=data)
            if response.status_code == 200:
                logging.info("Interfaz de Node-RED actualizada con los últimos datos.")
            else:
                logging.warning(f"Error al actualizar Node-RED: {response.status_code} - {response.text}")
        except Exception as e:
            logging.warning(f"No se pudo actualizar la interfaz de Node-RED. Detalle: {e}")

    def recibir_comandos(self):
        """
        Recibe comandos desde Node-RED para el control manual de los actuadores.
        Permite al operador activar o desactivar los actuadores.

        :return: Diccionario con los comandos recibidos.
        """
        try:
            # Realizar una solicitud GET para obtener los comandos manuales
            response = requests.get(f"{self.node_red_url}/manual_commands")
            if response.status_code == 200:
                commands = response.json()
                logging.info(f"Comandos recibidos desde Node-RED: {commands}")
                return commands
            else:
                logging.warning(f"No se pudieron obtener los comandos manuales desde Node-RED. Código: {response.status_code}")
                return {}
        except Exception as e:
            logging.error(f"Error al recibir comandos desde Node-RED: {e}")
            return {}

    def mostrar_sugerencias(self, sugerencias):
        """
        Envía sugerencias al operador a través de Node-RED.

        :param sugerencias: Diccionario con las sugerencias generadas por el sistema.
        """
        try:
            # Enviar las sugerencias mediante una solicitud POST
            response = requests.post(f"{self.node_red_url}/suggestions", json=sugerencias)
            if response.status_code == 200:
                logging.info("Sugerencias enviadas a Node-RED.")
            else:
                logging.warning(f"Error al enviar sugerencias a Node-RED: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Error al enviar sugerencias a Node-RED: {e}")
