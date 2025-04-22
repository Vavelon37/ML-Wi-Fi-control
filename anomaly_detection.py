from elasticsearch import Elasticsearch
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time
import requests
from datetime import datetime

es = Elasticsearch(
    [{"host": "localhost", "port": 9200, "scheme": "https"}],
    basic_auth=("elastic", "<PASSWORD>"),
    verify_certs=False  
)

def log_to_elasticsearch(index_name, level, message):
    try:
        document = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message
        }
        es.index(index=index_name, document=document)
    except Exception as e:
        print(f"Ошибка при логировании в Elasticsearch: {e}")

# Обучение Autoencoder
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(autoencoder, X_train):
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, shuffle=True, verbose=0)
    return autoencoder

def send_metric_to_prometheus(metric_name, value):
    url = "http://localhost:9091/metrics/job/anomaly_detection"
    payload = f"{metric_name} {value}\n" 
    try:
        response = requests.post(url, data=payload, headers={"Content-Type": "text/plain"})
        if response.status_code != 200:
            print(f"Ошибка отправки метрики в Prometheus: {response.status_code}, Ответ: {response.text}")
    except Exception as e:
        print(f"Ошибка отправки метрики в Prometheus: {e}")

def mitigate_attack(simulator):
    """Снижение нагрузки при атаке."""
    print("Атака обнаружена! Принимаются меры для снижения нагрузки...")
    simulator.base_traffic = max(100, simulator.base_traffic // 2)  
    simulator.base_latency = max(10, simulator.base_latency // 2)  

    log_to_elasticsearch("network_logs", "INFO", "Атака обнаружена и меры приняты.")

def restore_normal_state(simulator):
    """Восстановление нормального состояния сети."""
    print("Сеть восстановлена до нормального состояния.")
    simulator.base_traffic = 500  
    simulator.base_latency = 50

    log_to_elasticsearch("network_logs", "INFO", "Сеть восстановлена до нормального состояния.")

def detect_anomalies_realtime(autoencoder, scaler, threshold, simulator):
    previous_traffic = None
    latencies = []

    while True:
        network_state = simulator.generate_traffic()

        new_data = pd.DataFrame([[network_state["traffic"], network_state["latency"]]], columns=["traffic", "latency"])

        new_data_scaled = scaler.transform(new_data)

        reconstruction = autoencoder.predict(new_data_scaled)
        mse = np.mean(np.power(new_data_scaled - reconstruction, 2))

        is_anomaly = mse > threshold
        print(f"Трафик: {network_state['traffic']} МБ, Задержка: {network_state['latency']} мс, "
              f"Ошибка реконструкции: {mse:.4f}, Аномалия: {'Да' if is_anomaly else 'Нет'}")

        log_to_elasticsearch(
            index_name="network_logs",
            level="INFO" if not is_anomaly else "WARNING",
            message=f"Трафик: {network_state['traffic']} МБ, Задержка: {network_state['latency']} мс, Ошибка реконструкции: {mse:.4f}"
        )

        if is_anomaly:
            mitigate_attack(simulator)
        else:
            restore_normal_state(simulator)  

        send_metric_to_prometheus("network_traffic", network_state["traffic"])
        send_metric_to_prometheus("network_latency", network_state["latency"])
        send_metric_to_prometheus("anomaly_detected", int(is_anomaly))
        send_metric_to_prometheus("reconstruction_error", mse)

        if previous_traffic is not None:
            send_metric_to_prometheus("network_traffic_rate", abs(network_state["traffic"] - previous_traffic) / 2)
        send_metric_to_prometheus("network_latency_avg", sum(latencies) / len(latencies) if latencies else 0)

        previous_traffic = network_state["traffic"]
        latencies.append(network_state["latency"])
        if len(latencies) > 10:  
            latencies.pop(0)

        time.sleep(2)
# from elasticsearch import Elasticsearch
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import pandas as pd
# import time
# import requests

# es = Elasticsearch([{"host": "localhost", "port": 9200, "scheme": "http"}])

# def log_to_elasticsearch(index_name, document):
#     try:
#         es.index(index=index_name, document=document)
#     except Exception as e:
#         print(f"Ошибка при логировании в Elasticsearch: {e}")

# def build_autoencoder(input_dim):
#     input_layer = Input(shape=(input_dim,))
#     encoded = Dense(64, activation='relu')(input_layer)
#     encoded = Dense(32, activation='relu')(encoded)
#     decoded = Dense(64, activation='relu')(encoded)
#     decoded = Dense(input_dim, activation='sigmoid')(decoded)
#     autoencoder = Model(inputs=input_layer, outputs=decoded)
#     autoencoder.compile(optimizer='adam', loss='mse')
#     return autoencoder

# def train_autoencoder(autoencoder, X_train):
#     autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, shuffle=True, verbose=0)
#     return autoencoder

# def send_metric_to_prometheus(metric_name, value):
#     url = "http://localhost:9091/metrics/job/anomaly_detection"
#     payload = f"{metric_name} {value}\n" 
#     try:
#         response = requests.post(url, data=payload, headers={"Content-Type": "text/plain"})
#         if response.status_code != 200:
#             print(f"Ошибка отправки метрики в Prometheus: {response.status_code}, Ответ: {response.text}")
#     except Exception as e:
#         print(f"Ошибка отправки метрики в Prometheus: {e}")

# def mitigate_attack(simulator):
#     """Снижение нагрузки при атаке."""
#     print("Атака обнаружена! Принимаются меры для снижения нагрузки...")
#     simulator.base_traffic = max(100, simulator.base_traffic // 2)  
#     simulator.base_latency = max(10, simulator.base_latency // 2)  

#     log_to_elasticsearch("network_logs", {
#         "event": "attack_mitigated",
#         "timestamp": pd.Timestamp.now().isoformat(),
#         "new_traffic": simulator.base_traffic,
#         "new_latency": simulator.base_latency
#     })

# def restore_normal_state(simulator):
#     """Восстановление нормального состояния сети."""
#     print("Сеть восстановлена до нормального состояния.")
#     simulator.base_traffic = 500  
#     simulator.base_latency = 50

#     # Логирование события в Elasticsearch
#     log_to_elasticsearch("network_logs", {
#         "event": "normal_state_restored",
#         "timestamp": pd.Timestamp.now().isoformat(),
#         "traffic": simulator.base_traffic,
#         "latency": simulator.base_latency
#     })

# def detect_anomalies_realtime(autoencoder, scaler, threshold, simulator):
#     previous_traffic = None
#     latencies = []

#     while True:
#         network_state = simulator.generate_traffic()

#         new_data = pd.DataFrame([[network_state["traffic"], network_state["latency"]]], columns=["traffic", "latency"])

#         new_data_scaled = scaler.transform(new_data)

#         reconstruction = autoencoder.predict(new_data_scaled)
#         mse = np.mean(np.power(new_data_scaled - reconstruction, 2))

#         is_anomaly = mse > threshold
#         print(f"Трафик: {network_state['traffic']} МБ, Задержка: {network_state['latency']} мс, "
#               f"Ошибка реконструкции: {mse:.4f}, Аномалия: {'Да' if is_anomaly else 'Нет'}")

#         log_to_elasticsearch("network_logs", {
#             "event": "network_state",
#             "timestamp": network_state["timestamp"].isoformat(),
#             "traffic": network_state["traffic"],
#             "latency": network_state["latency"],
#             "is_anomaly": is_anomaly
#         })

#         if is_anomaly:
#             mitigate_attack(simulator)  
#         else:
#             restore_normal_state(simulator)  

#         send_metric_to_prometheus("network_traffic", network_state["traffic"])
#         send_metric_to_prometheus("network_latency", network_state["latency"])
#         send_metric_to_prometheus("anomaly_detected", int(is_anomaly))
#         send_metric_to_prometheus("reconstruction_error", mse)

#         if previous_traffic is not None:
#             send_metric_to_prometheus("network_traffic_rate", abs(network_state["traffic"] - previous_traffic) / 2)
#         send_metric_to_prometheus("network_latency_avg", sum(latencies) / len(latencies) if latencies else 0)

#         previous_traffic = network_state["traffic"]
#         latencies.append(network_state["latency"])
#         if len(latencies) > 10: 
#             latencies.pop(0)

#         time.sleep(2)