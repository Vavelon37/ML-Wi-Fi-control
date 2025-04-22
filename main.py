from network_simulator import NetworkSimulator
from anomaly_detection import build_autoencoder, train_autoencoder, detect_anomalies_realtime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    simulator = NetworkSimulator()

    data = []
    for _ in range(100):
        state = simulator.generate_traffic()
        data.append([state["traffic"], state["latency"]])

    df = pd.DataFrame(data, columns=["traffic", "latency"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    X_train, X_val = X_scaled[:80], X_scaled[80:]

    autoencoder = build_autoencoder(X_train.shape[1])
    autoencoder = train_autoencoder(autoencoder, X_train)

    reconstructions = autoencoder.predict(X_val)
    mse = np.mean(np.power(X_val - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)

    print("Запуск обнаружения аномалий в реальном времени...")
    detect_anomalies_realtime(autoencoder, scaler, threshold, simulator)
# from network_simulator import NetworkSimulator
# from anomaly_detection import build_autoencoder, train_autoencoder, detect_anomalies_realtime
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# if __name__ == "__main__":
#     simulator = NetworkSimulator()

#     data = []
#     for _ in range(100):  
#         state = simulator.generate_traffic()
#         data.append([state["traffic"], state["latency"]])

#     df = pd.DataFrame(data, columns=["traffic", "latency"])
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df)

#     X_train, X_val = X_scaled[:80], X_scaled[80:]

#     autoencoder = build_autoencoder(X_train.shape[1])
#     autoencoder = train_autoencoder(autoencoder, X_train)

#     reconstructions = autoencoder.predict(X_val)
#     mse = np.mean(np.power(X_val - reconstructions, 2), axis=1)
#     threshold = np.percentile(mse, 95)

#     print("Запуск обнаружения аномалий в реальном времени...")
#     detect_anomalies_realtime(autoencoder, scaler, threshold, simulator)