import requests

# Change this if your API is on a different host/port
API_URL = "http://localhost:5001/refresh_embeddings"

# Example payload: replace with your actual KB names
payload = {
    "kb_names": ["core_rules", "adversaries"]
}

try:
    response = requests.post(API_URL, json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Request failed:", e)