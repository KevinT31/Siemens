# controller.py

import time
import logging
import threading
import os
import csv
import sys
from datetime import datetime

from sensors import SoilSensor, LevelSensor, FlowSensor
from actuators import PumpControl, ValveControl
from signal_conditioning import SignalConditioning
from decision_engine import DecisionEngine
from model_training import SistemaRiegoInteligente  # Importar la clase correcta
from gui import NodeRedInterface

# Configuración del logging para registrar eventos y errores
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class ControladorSistemaRiego:
    """
    Clase principal que controla el sistema de riego automatizado.
    """

    def __init__(self):
        # Inicialización de sensores
        self.soil_sensor = SoilSensor(address=1)

        # Sensor de nivel de agua (Ultrasonido US-100)
        self.level_sensor = LevelSensor(
            trig_pin=18,  # GPIO18 para TRIG
            echo_pin=24,  # GPIO24 para ECHO
            calibration_params={
                'tank_height': 200  # Altura del tanque en cm
            }
        )

        # Inicializar sensor de flujo
        self.flow_sensor = FlowSensor(gpio_pin=17, calibration_params={
            'factor': 5.5,
            'tolerance': 0.2
        })

        # Acondicionamiento de señal
        self.signal_conditioning = SignalConditioning()

        # Motor de toma de decisiones (modelo de Machine Learning)
        self.decision_engine = DecisionEngine(model_path='modelo_actualizado.pkl')

        # Entrenamiento y actualización del modelo
        self.model_training = SistemaRiegoInteligente(
            model_path='modelo_actualizado.pkl',
            data_path='data/decision_data.csv'  # Aseguramos que coincide con el archivo de datos
        )

        # Interfaz gráfica (GUI) en Node-RED
        self.gui = NodeRedInterface()

        # Parámetros del sistema
        self.data_collection_frequency = 60  # Frecuencia de recolección de datos en segundos (cada minuto)
        self.sensor_data = []

        # Definir pines GPIO para los actuadores
        GPIO_PINS = {
            'bomba': 25,
            'valvula_suministro': 24,
        }

        # Inicialización de actuadores
        self.pump_control = PumpControl(GPIO_PINS['bomba'])
        self.valve_control = ValveControl({
            'suministro_alternativo': GPIO_PINS['valvula_suministro']
        })

        # Estados de los actuadores
        self.actuator_states = {
            'bomba': False,
            'valvula_suministro': False,
        }

        # Modo de simulación
        self.simulation_mode = sys.platform == "win32"

        # Definir el modo de control (por ejemplo, 'automatico' o 'manual')
        self.control_mode = 'automatico'

        logging.info("Controlador del Sistema de Riego inicializado.")

        # Entrenar el modelo al iniciar el controlador
        self._entrenar_modelo()

    def _entrenar_modelo(self):
        """
        Entrena el modelo de Machine Learning utilizando los datos disponibles.
        """
        logging.info("Iniciando entrenamiento del modelo de Machine Learning...")
        self.model_training.entrenar_modelo_local()
        # Actualizar el modelo en el motor de decisiones
        self.decision_engine.actualizar_modelo(self.model_training.model)
        logging.info("Modelo de Machine Learning entrenado y actualizado en el motor de decisiones.")

    def iniciar(self):
        """
        Método principal que inicia el ciclo de control del sistema de riego.
        """
        logging.info("Iniciando el sistema de riego...")
        while True:
            try:
                # 1. Adquisición de datos de sensores
                sensor_values = self._leer_sensores()

                # 2. Procesar los datos con la lógica de toma de decisiones
                decision = self._tomar_decision(sensor_values)

                # 3. Accionar los actuadores según la decisión
                self._accionar_actuadores(decision)

                # 4. Detectar anomalías en el flujo de agua
                self._verificar_anomalias(sensor_values)

                # 5. Actualizar la interfaz gráfica con los nuevos datos y estado del sistema
                self.gui.actualizar_interfaz(sensor_values, self.control_mode)

                # 6. Esperar hasta la próxima iteración
                time.sleep(self.data_collection_frequency)  # Espera de 60 segundos (1 minuto)

            except Exception as e:
                logging.error(f"Error en el ciclo principal: {e}")
                time.sleep(self.data_collection_frequency)  # Espera antes de reintentar

    def _leer_sensores(self):
        """
        Adquiere los datos de todos los sensores.
        """
        logging.info("Leyendo datos de sensores...")
        try:
            # Lectura del sensor de suelo
            soil_values = self.soil_sensor.leer()
            if soil_values is None:
                raise Exception("Error al leer el sensor de suelo")

            humidity = soil_values['humidity']
            temperature = soil_values['temperature']

            # Lectura del sensor de nivel de agua
            water_level = self.level_sensor.leer()
            if water_level is None:
                raise Exception("Error al leer el sensor de nivel de agua")

            # Lectura del sensor de flujo
            flow_rate = self.flow_sensor.leer()
            if flow_rate is None:
                raise Exception("Error al leer el sensor de flujo")

            # Acondicionamiento de señal
            humidity = self.signal_conditioning.acondicionar_humedad(humidity)
            temperature = self.signal_conditioning.acondicionar_temperatura(temperature)
            water_level = self.signal_conditioning.acondicionar_nivel(water_level)

            # Obtener el timestamp actual
            timestamp = time.time()

            # Agrupar datos en un diccionario
            sensor_values = {
                'timestamp': timestamp,
                'humidity': round(humidity, 1),
                'temperature': round(temperature, 1),
                'water_level': round(water_level, 1),
                'flow_rate': round(flow_rate, 1)  # Ajustado a 1 decimal
            }

            logging.info(f"Datos adquiridos: {sensor_values}")

            # Almacenar datos localmente
            self.sensor_data.append(sensor_values)

            # Guardar los datos en el archivo sensors_data.csv
            self._guardar_datos_csv(sensor_values)

            return sensor_values

        except Exception as e:
            logging.error(f"Error al leer sensores: {e}")
            raise

    def _verificar_anomalias(self, sensor_values):
        """
        Verifica si hay anomalías en el flujo de agua, indicando posibles fugas o bloqueos.
        """
        try:
            # Determinar si se espera flujo de agua basado en el estado de la bomba
            bomba_activa = self.actuator_states.get('bomba', False)

            expected_flow = None
            if bomba_activa:
                # Si la bomba está activa, se espera un flujo normal
                expected_flow = self.flow_sensor.calibration_params.get('expected_flow', 30)  # Valor promedio esperado
            else:
                # Si la bomba está desactivada, no debería haber flujo
                expected_flow = 1

            # Detectar fallos en el flujo
            flow_anomaly = self.flow_sensor.detectar_fallo(expected_flow)
            if flow_anomaly:
                # Ya se ha registrado el error dentro de detectar_fallo()
                pass

        except Exception as e:
            logging.error(f"Error al verificar anomalías de flujo: {e}")
            raise

    def _guardar_datos_csv(self, sensor_values):
        """
        Guarda los datos de los sensores en el archivo data/sensors_data.csv.
        """
        # Ruta del archivo CSV
        csv_file = 'data/sensors_data.csv'
        file_exists = os.path.isfile(csv_file)

        # Asegurarse de que el directorio 'data' existe
        os.makedirs('data', exist_ok=True)

        # Convertir timestamp a formato legible sin modificar sensor_values
        timestamp_str = datetime.fromtimestamp(sensor_values['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

        # Crear una copia de sensor_values para escribir al CSV
        sensor_values_csv = {
            'timestamp': timestamp_str,
            'humidity': sensor_values['humidity'],
            'temperature': sensor_values['temperature'],
            'water_level': sensor_values['water_level'],
            'flow_rate': sensor_values['flow_rate']
        }

        # Campos del CSV
        campos = ['timestamp', 'humidity', 'temperature', 'water_level', 'flow_rate']

        try:
            # Escribir en el CSV usando sensor_values_csv
            with open(csv_file, mode='a', newline='') as archivo_csv:
                escritor = csv.DictWriter(archivo_csv, fieldnames=campos)
                if not file_exists:
                    escritor.writeheader()
                escritor.writerow(sensor_values_csv)
            logging.info("Datos de sensores guardados en data/sensors_data.csv")
        except Exception as e:
            logging.error(f"Error al guardar datos en CSV: {e}")
            raise

    def _tomar_decision(self, sensor_values):
        """
        Toma decisiones basadas en los datos de los sensores y la lógica definida.
        """
        logging.info("Procesando datos y tomando decisiones...")
        try:
            # Evaluar umbrales críticos
            umbrales_ok = self._verificar_umbrales(sensor_values)

            if not umbrales_ok:
                logging.warning("Datos fuera de umbrales permitidos. Tomando acciones de emergencia.")
                decision = self._acciones_emergencia(sensor_values)
            else:
                # Utilizar el motor de decisiones (Machine Learning) para tomar decisiones
                decision = self.decision_engine.evaluar(sensor_values)

                # Lógica adicional para el suministro alternativo
                if decision.get('activar_bomba'):
                    if sensor_values['water_level'] < 20:
                        decision['abrir_valvula_suministro'] = True
                        logging.info("Nivel de agua bajo y se requiere riego. Abriendo suministro alternativo.")
                    else:
                        decision['abrir_valvula_suministro'] = False
                        logging.info("Nivel de agua suficiente para riego. Suministro alternativo cerrado.")
                else:
                    # Si no se va a activar la bomba, no es necesario abrir el suministro alternativo
                    decision['abrir_valvula_suministro'] = False
                    logging.info("No se requiere riego. Suministro alternativo cerrado.")

            # Guardar la decisión en el archivo decision_data.csv
            self._guardar_decision_csv(sensor_values, decision)

            return decision

        except Exception as e:
            logging.error(f"Error al tomar decisiones: {e}")
            raise

    def _accionar_actuadores(self, decision):
        """
        Controla los actuadores según la decisión tomada.
        """
        try:
            # Control de la bomba hidráulica
            if decision.get('activar_bomba'):
                self.pump_control.activar()
                self.actuator_states['bomba'] = True
            else:
                self.pump_control.desactivar()
                self.actuator_states['bomba'] = False

            # Control de la válvula de suministro alternativo
            if decision.get('abrir_valvula_suministro'):
                self.valve_control.abrir_valvula('suministro_alternativo')
                self.actuator_states['valvula_suministro'] = True
            else:
                self.valve_control.cerrar_valvula('suministro_alternativo')
                self.actuator_states['valvula_suministro'] = False

            logging.info(f"Acciones ejecutadas: {decision}")

        except Exception as e:
            logging.error(f"Error al accionar actuadores: {e}")
            raise

    def _verificar_umbrales(self, sensor_values):
        """
        Verifica si los datos de los sensores están dentro de los umbrales permitidos.
        Retorna True si todos los valores están dentro de los límites, False de lo contrario.
        """
        UMBRALES = {
            'humidity': {'min': 20, 'max': 80},
            'temperature': {'min': -10, 'max': 35},
            'water_level': {'min': 0, 'max': 100},  # Ajustamos los límites según corresponda
            'flow_rate': {'min': 0.0, 'max': 60.0}
        }

        umbrales_ok = True
        for sensor, limits in UMBRALES.items():
            value = sensor_values.get(sensor)
            if value is None:
                continue
            if not (limits['min'] <= value <= limits['max']):
                logging.warning(f"{sensor} fuera de umbrales: {value} (Límites: {limits})")
                umbrales_ok = False

        return umbrales_ok

    def _acciones_emergencia(self, sensor_values):
        """
        Toma acciones correctivas inmediatas cuando los datos están fuera de los umbrales permitidos.
        """
        # Inicializar decisiones con valores por defecto
        decision = {
            'activar_bomba': False,
            'abrir_valvula_suministro': False
        }

        try:
            # Acciones de emergencia para humedad
            if sensor_values['humidity'] < 20:
                decision['activar_bomba'] = True
                logging.info("Emergencia: Humedad muy baja. Activando riego.")
            elif sensor_values['humidity'] > 80:
                decision['activar_bomba'] = False
                logging.info("Emergencia: Humedad muy alta. Desactivando riego.")

            # Acciones de emergencia para temperatura
            if sensor_values['temperature'] > 35:
                # Temperatura muy alta
                decision['activar_bomba'] = True  # Activamos riego para enfriar el suelo
                logging.info("Emergencia: Temperatura muy alta. Activando riego para enfriar el suelo.")
            elif sensor_values['temperature'] < -10:
                # Temperatura muy baja
                decision['activar_bomba'] = False  # Reducimos riego para evitar enfriar más el suelo
                logging.info("Emergencia: Temperatura muy baja. Desactivando riego para evitar enfriar el suelo.")

            # Acciones de emergencia para nivel de agua y riego
            if 'water_level' in sensor_values:
                if sensor_values['water_level'] < 20:
                    if decision['activar_bomba']:
                        decision['abrir_valvula_suministro'] = True
                        logging.info("Emergencia: Nivel de agua bajo y se requiere riego. Abriendo suministro alternativo.")
                    else:
                        decision['abrir_valvula_suministro'] = False
                        logging.info("Nivel de agua bajo pero no se requiere riego. Suministro alternativo cerrado.")
                else:
                    decision['abrir_valvula_suministro'] = False
                    logging.info("Nivel de agua adecuado. Suministro alternativo cerrado.")

            return decision

        except Exception as e:
            logging.error(f"Error en acciones de emergencia: {e}")
            raise

    def _guardar_decision_csv(self, sensor_values, decision):
        """
        Guarda las decisiones tomadas en el archivo data/decision_data.csv junto con los valores de los sensores.
        """
        decision_csv = 'data/decision_data.csv'  # Nombre del archivo
        file_exists = os.path.isfile(decision_csv)

        # Asegurarse de que el directorio 'data' existe
        os.makedirs('data', exist_ok=True)

        # Convertir timestamp a formato legible sin modificar sensor_values
        timestamp_str = datetime.fromtimestamp(sensor_values['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

        # Combinar sensor_values y decision en un solo diccionario
        data_csv = {
            'timestamp': timestamp_str,
            'humidity': sensor_values['humidity'],
            'temperature': sensor_values['temperature'],
            'water_level': sensor_values['water_level'],
            'flow_rate': sensor_values['flow_rate'],
            'activar_bomba': int(decision.get('activar_bomba', False)),
            'abrir_valvula_suministro': int(decision.get('abrir_valvula_suministro', False))
        }

        # Campos del CSV
        campos = ['timestamp', 'humidity', 'temperature', 'water_level', 'flow_rate',
                  'activar_bomba', 'abrir_valvula_suministro']

        try:
            # Escribir en el CSV usando data_csv
            with open(decision_csv, mode='a', newline='') as archivo_csv:
                escritor = csv.DictWriter(archivo_csv, fieldnames=campos)
                if not file_exists:
                    escritor.writeheader()
                escritor.writerow(data_csv)

            logging.info("Decisión guardada en data/decision_data.csv.")

        except Exception as e:
            logging.error(f"Error al guardar decisión en CSV: {e}")
            raise
