import random
import pandas as pd

class NetworkSimulator:
    def __init__(self):
        self.base_traffic = 500  
        self.base_latency = 50   
        self.noise_level = 10    
        self.anomaly_probability = 0.05 

    def generate_traffic(self):
        """Генерация текущего состояния сети."""
        traffic = self.base_traffic + random.uniform(-self.noise_level, self.noise_level) * self.base_traffic / 100
        latency = self.base_latency + random.uniform(-self.noise_level, self.noise_level) * self.base_latency / 100

        if random.random() < self.anomaly_probability:
            traffic *= random.uniform(2, 5)  
            latency *= random.uniform(2, 5)  

        return {
            "traffic": round(traffic, 2),
            "latency": round(latency, 2),
            "timestamp": pd.Timestamp.now()
        }
# import random
# import pandas as pd

# class NetworkSimulator:
#     def __init__(self):
#         self.base_traffic = 500 
#         self.base_latency = 50   
#         self.noise_level = 10   
#         self.anomaly_probability = 0.05  

#     def generate_traffic(self):
#         """Генерация текущего состояния сети."""
#         traffic = self.base_traffic + random.uniform(-self.noise_level, self.noise_level) * self.base_traffic / 100
#         latency = self.base_latency + random.uniform(-self.noise_level, self.noise_level) * self.base_latency / 100

#         
#         if random.random() < self.anomaly_probability:
#             traffic *= random.uniform(2, 5) 
#             latency *= random.uniform(2, 5) 

#         return {
#             "traffic": round(traffic, 2),
#             "latency": round(latency, 2),
#             "timestamp": pd.Timestamp.now()
#           }