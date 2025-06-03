import requests

url = "http://localhost:8000/predict"

with open("test_image.jpeg", "rb") as f:
    files = {"image": ("test_image.jpeg", f, "image/jpeg")}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())