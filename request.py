import requests

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
)

print(f"Status code: {response.status_code}")
print(f"Response text: {response.text}")
print(f"Response headers: {response.headers}")

if response.status_code == 200:
    try:
        print(f"JSON response: {response.json()}")
    except:
        print("Failed to parse JSON")
else:
    print("Request failed")