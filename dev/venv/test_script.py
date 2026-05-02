import csv
import json
import random

# --- CONFIGURATION ---
users = [
    {"email": "aloka@gmail.com", "password": "abcdef"},
    {"email": "sandali@gmail.com", "password": "abcdef"},
    {"email": "user@example.com", "password": "newSecret123"},
    {"email": "yonal@gmail.com", "password": "654321"},
    {"email": "chamod99@gmail.com", "password": "12345"},
    {"email": "kweerasinghe@gmail.com", "password": "123456"},
    {"email": "dilshani123@gmail.com", "password": "123456"},
    {"email": "sandaliliyanage@ieee.com", "password": "123456"}
]

# Category and Sub-category mapping
cat_config = {
    1: list(range(10, 17)),
    2: list(range(20, 26)),
    3: list(range(30, 38))
}

def generate_session(user, writer):
    rows_added = 0
    
    # 1. Login Logic (Success or Failure Sequence)
    is_failed_session = random.random() < 0.2  # 20% අවස්ථාවකදී login fail වෙයි
    
    if is_failed_session:
        # Failed Logins (2 or 3 times)
        fail_count = random.randint(2, 3)
        for _ in range(fail_count):
            writer.writerow(["login1", json.dumps({"email": user["email"], "password": "wrong_password"})])
            rows_added += 1
        
        # Next: Forgot Password
        writer.writerow(["forgot-password", json.dumps({
            "email": user["email"], 
            "password": user["password"], 
            "confirmPassword": user["password"]
        })])
        rows_added += 1
        
        # Next: Successful Login1
        writer.writerow(["login1", json.dumps({"email": user["email"], "password": user["password"]})])
        rows_added += 1
    else:
        # Normal Successful Login1 or Signup
        func = "signup" if random.random() < 0.1 else "login1"
        writer.writerow([func, json.dumps({"email": user["email"], "password": user["password"]})])
        rows_added += 1

    # 2. Activity Logic (Category -> Sub-category -> Question)
    num_activities = random.randint(1, 3)
    for _ in range(num_activities):
        cat_id = random.randint(1, 3)
        sub_id = random.choice(cat_config[cat_id])
        
        # Step A: Select Category
        writer.writerow(["select-category", json.dumps({"categoryId": cat_id})])
        rows_added += 1
        
        # Step B: Select Sub-Category
        writer.writerow(["select-sub-category", json.dumps({"categoryId": cat_id, "subCategoryId": sub_id})])
        rows_added += 1
        
        # Step C: Optional Select Question (50% chance)
        if random.random() > 0.5:
            q_id = f"{cat_id}{sub_id}{random.randint(0, 4)}"
            writer.writerow(["select-question", json.dumps({
                "categoryId": cat_id, 
                "subCategoryId": sub_id, 
                "questionId": int(q_id)
            })])
            rows_added += 1

    # 3. Logout
    writer.writerow(["logout", json.dumps({"email": user["email"]})])
    rows_added += 1
    
    return rows_added

# --- MAIN EXECUTION ---
with open('test_data2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["function", "payload"])
    
    total_rows = 0
    while total_rows < 200:
        current_user = random.choice(users)
        added = generate_session(current_user, writer)
        total_rows += added

print(f"✅ Sequential User Sessions සහිත test_data.csv සාදා නිමයි! මුළු පේළි ගණන: {total_rows}")