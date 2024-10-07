import requests

url = "http://127.0.0.1:9527" + "/register_worker"
data = {
    "worker_name": "",
    "check_heart_beat": True,
    "worker_status": "",
    "multimodal": " ",
}
r = requests.post(url, json=data)
print(r.json())
