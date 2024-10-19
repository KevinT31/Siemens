# model_training.py

import os
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from datetime import timedelta

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class SistemaRiegoInteligente:
    def __init__(self, model_path="modelo_actualizado.pkl", data_path="data/decision_data.csv", target="activar_bomba"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.target = target
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def cargar_datos(self):
        try:
            if not os.path.exists(self.data_path):
                logger.error(f"El archivo de datos '{self.data_path}' no existe.")
                return None, None
            data = pd.read_csv(self.data_path)
            logger.info("Datos cargados exitosamente para el entrenamiento.")

            if self.target not in data.columns:
                logger.error(f"La columna '{self.target}' no existe en los datos.")
                return None, None

            # Seleccionar características y etiqueta
            X = data[['humidity', 'temperature', 'water_level', 'flow_rate']]
            y = data[self.target]

            # Manejar valores faltantes
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.dropna(inplace=True)
            y = y[X.index]  # Alinear 'y' con 'X' después de eliminar filas con valores faltantes

            # Eliminar outliers utilizando Isolation Forest
            iso = IsolationForest(contamination=0.05, random_state=42)
            yhat = iso.fit_predict(X)
            mask = yhat != -1
            X, y = X[mask], y[mask]
            logger.info(f"Datos después de eliminar outliers: {X.shape}")

            return X, y
        except Exception as e:
            logger.error(f"Error al cargar y preparar los datos: {e}")
            return None, None

    def entrenar_modelo_local(self):
        X, y = self.cargar_datos()
        if X is None or y is None:
            logger.error("No se pudo entrenar el modelo debido a errores en los datos.")
            return None
        try:
            # Verificar el balance de clases
            clase_counts = y.value_counts()
            logger.info(f"Distribución de clases antes del balanceo:\n{clase_counts}")

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            # Aplicar SMOTE para balancear las clases en el conjunto de entrenamiento
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            # Verificar el balance de clases después de SMOTE
            clase_counts_res = pd.Series(y_train_res).value_counts()
            logger.info(f"Distribución de clases después de SMOTE:\n{clase_counts_res}")

            # Escalar las características usando MinMaxScaler ajustado solo con X_train_res
            scaler = MinMaxScaler()
            X_train_res = scaler.fit_transform(X_train_res)
            X_test = scaler.transform(X_test)

            # Ajustar scale_pos_weight para XGBoost
            scale_pos_weight = sum(y_train_res == 0) / sum(y_train_res == 1)

            # Definir los hiperparámetros a ajustar (simplificados)
            param_dist = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [1, 1.5, 2],
                'scale_pos_weight': [scale_pos_weight]
            }

            # Configurar búsqueda aleatoria de hiperparámetros
            random_search = RandomizedSearchCV(
                XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                param_distributions=param_dist,
                n_iter=30,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            random_search.fit(X_train_res, y_train_res)

            # Guardar el mejor modelo
            self.model = random_search.best_estimator_
            logger.info(f"Mejores hiperparámetros encontrados:\n{random_search.best_params_}")

            # Realizar predicciones
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred)
            logger.info(f"Reporte de clasificación:\n{report}")

            # Almacenar X_test, y_test y y_pred para evaluación posterior
            self.X_test = X_test
            self.y_test = y_test
            self.y_pred = y_pred

            # Guardar el modelo entrenado
            self.guardar_modelo()
        except Exception as e:
            logger.error(f"Error durante el entrenamiento del modelo: {e}")

    def guardar_modelo(self):
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Modelo entrenado guardado en {self.model_path}")
        except Exception as e:
            logger.error(f"Error al guardar el modelo: {e}")

    def cargar_modelo(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Modelo cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            self.model = None

    def predecir(self, X):
        if self.model is None:
            logger.error("No hay un modelo cargado para realizar predicciones.")
            return None
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Error al realizar predicciones: {e}")
            return None

# Verificar si el directorio 'data' existe
os.makedirs('data', exist_ok=True)

# Ruta al archivo de datos
data_file_path = 'data/decision_data.csv'

# Verificar si 'data/decision_data.csv' existe
if not os.path.exists(data_file_path):
    logger.error(f"El archivo '{data_file_path}' no existe. No se pueden cargar datos para entrenamiento.")
    exit()

# Cargar los datos existentes
df = pd.read_csv(data_file_path)

# Asegurarse de que 'timestamp' esté en formato datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Verificar que las columnas necesarias existan en el DataFrame
required_columns = ['timestamp', 'humidity', 'temperature', 'water_level', 'flow_rate',
                    'activar_bomba', 'abrir_valvula_suministro']

for col in required_columns:
    if col not in df.columns:
        logger.error(f"La columna '{col}' no existe en los datos.")
        exit()

# Verificar si se necesitan más datos para el entrenamiento
if df.shape[0] < 1000:
    logger.info("Generando datos sintéticos para complementar el entrenamiento.")
    # Generar datos sintéticos para complementar hasta 1000 registros
    num_synthetic = 1000 - df.shape[0]

    # Obtener el último timestamp
    last_timestamp = df['timestamp'].iloc[-1]

    # Lista para almacenar nuevos datos
    new_data_list = []

    for i in range(num_synthetic):
        # Generar nuevo timestamp incrementando en 1 minuto
        new_timestamp = last_timestamp + timedelta(minutes=1)
        last_timestamp = new_timestamp

        # Generar valores sintéticos con correlaciones realistas
        base_temperature = np.random.normal(25, 5)
        temperature_noise = np.random.normal(0, 2)
        new_temperature = round(base_temperature + temperature_noise, 1)

        humidity_noise = np.random.normal(0, 5)
        new_humidity = round(max(0, 100 - new_temperature * np.random.uniform(1.5, 2.5) + humidity_noise), 1)

        base_water_level = np.random.normal(75, 10)
        water_level_noise = np.random.normal(0, 5)
        new_water_level = round(base_water_level + water_level_noise, 1)

        flow_rate_noise = np.random.normal(0, 1)
        new_flow_rate = round(max(0, (new_water_level / 100) * 10 + flow_rate_noise), 1)

        # Decisiones basadas en humedad y temperatura
        probabilidad_activar_bomba = 0.0

        if new_humidity < 40 and new_temperature > 30:
            probabilidad_activar_bomba = 0.9
        elif new_humidity < 40 or new_temperature > 30:
            probabilidad_activar_bomba = 0.7
        else:
            probabilidad_activar_bomba = 0.1

        new_activar_bomba = np.random.choice([1, 0], p=[probabilidad_activar_bomba, 1 - probabilidad_activar_bomba])

        # Ajustar apertura de válvula de suministro alternativo
        if new_water_level < 30:
            new_abrir_valvula_suministro = 1
        else:
            new_abrir_valvula_suministro = 0

        new_data_list.append({
            'timestamp': new_timestamp,
            'humidity': new_humidity,
            'temperature': new_temperature,
            'water_level': new_water_level,
            'flow_rate': new_flow_rate,
            'activar_bomba': new_activar_bomba,
            'abrir_valvula_suministro': new_abrir_valvula_suministro
        })

    # Convertir la lista de nuevos datos en DataFrame
    new_data_df = pd.DataFrame(new_data_list)

    # Concatenar los nuevos datos al DataFrame original
    df = pd.concat([df, new_data_df], ignore_index=True)

    # Asegurarse de que las columnas booleanas son enteros
    boolean_columns = ['activar_bomba', 'abrir_valvula_suministro']
    df[boolean_columns] = df[boolean_columns].astype(int)

    # Guardar los datos aumentados
    df.to_csv(data_file_path, index=False)
    logger.info(f"Datos sintéticos agregados y guardados en '{data_file_path}'.")

# Entrenamiento del modelo
training = SistemaRiegoInteligente(data_path=data_file_path, target='activar_bomba')
training.entrenar_modelo_local()

# Matriz de confusión
if training.y_test is not None and training.y_pred is not None:
    cm = confusion_matrix(training.y_test, training.y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    cm_display.plot()
    plt.show()
else:
    logger.error("No se pudo generar la matriz de confusión debido a que no hay datos de prueba o predicciones.")

accuracy = accuracy_score(training.y_test, training.y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")