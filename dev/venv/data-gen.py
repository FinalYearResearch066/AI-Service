import pandas as pd
import numpy as np
import random

def generate_final_dataset(n_samples=10000):
    functions = ["login", "logout", "signup", "reset_password", "search_category", "fetch_correct_code", "fetch_questions", "edit_profile"]
    users = [f"U{i:03d}" for i in range(1, 101)] # Users 100
    
    data = []
    avg_warm_execution = 60 # ms (Standard execution time)

    for i in range(n_samples):
        func = random.choice(functions)
        user = random.choice(users)
        hour = np.random.randint(0, 24)
        day = np.random.randint(0, 7)

        # Cold start logic based on intent/context
        is_cold = False
        if (hour < 6 or hour > 22) or (func in ["signup", "edit_profile"]):
            if random.random() > 0.4: # 60% chance of cold start in these conditions
                is_cold = True
        
        if is_cold:
            # Total duration recorded by Prometheus (includes startup delay)
            total_duration = avg_warm_execution + np.random.randint(350, 950)
        else:
            # Warm start (container already exists)
            total_duration = avg_warm_execution + np.random.randint(-15, 30)

        # Derived Cold Start Latency = Total - Avg Warm Time
        derived_cold_start = max(0, total_duration - avg_warm_execution)
        
        data.append([func, user, hour, day, total_duration, derived_cold_start])

    df = pd.DataFrame(data, columns=['function', 'user', 'hour', 'day', 'total_duration', 'derived_cold_start'])
    df.to_csv('openfaas_research_data.csv', index=False)
    print("✅ Dataset saved as 'openfaas_research_data.csv' (10,000 rows)")

if __name__ == "__main__":
    generate_final_dataset()