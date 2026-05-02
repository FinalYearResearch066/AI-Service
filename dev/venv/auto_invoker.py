import requests
import time
import random
from datetime import datetime

# --- CONFIGURATION ---
SCALER_URL = "http://127.0.0.1:5000/invoke"
FUNCTIONS = ["signup","login1","forgot-password","select-category","select-sub-category","select-question","logout"]
USERS = list(range(1, 9))  # 1 සිට 8 දක්වා පරිශීලකයින්

def invoke_functions():
    print(f"--- Cycle started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # පවතින functions අතරින් අහඹු ලෙස කිහිපයක් තෝරා ගැනීම හෝ සියල්ල පිළිවෙලින් යැවීම
    # මෙහිදී functions 7 ම විවිධ අහඹු පරිශීලකයින් හරහා invoke කරයි
    sampled_functions = list(FUNCTIONS)
    random.shuffle(sampled_functions)

    for func in sampled_functions:
        user_id = random.choice(USERS)
        payload = {
            "function_name": func,
            "user_id": user_id  # ඔබේ පද්ධතියේ user ID අවශ්‍ය නම් පමණක්
        }
        
        try:
            response = requests.post(SCALER_URL, json=payload)
            if response.status_code == 200:
                print(f"[SUCCESS] Function: {func} | User: {user_id} | Response: {response.json()}")
            else:
                print(f"[FAILED] Function: {func} | Status Code: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Could not connect to Scaler: {e}")
        
        # එක් function එකක් සහ තව එකක් අතර කුඩා ප්‍රමාදයක් (උදා: තත්පර 2)
        time.sleep(2)

    print(f"--- Cycle finished. Waiting for 1 hour... ---")

if __name__ == "__main__":
    while True:
        invoke_functions()
        # පැයක කාලයක් (තත්පර 3600) බලා සිටීම
        time.sleep(3600)