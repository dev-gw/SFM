import socket
import time

HOST = '192.168.0.114'
PORT = 7777

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

client_socket.connect((HOST, PORT))



# 키보드로 입력한 문자열을 서버로 전송하고
# 서버에서 에코되어 돌아오는 메시지를 받으면 화면에 출력합니다.
# quit를 입력할 때 까지 반복합니다.
i = 0
o = 0
c = 125
w = 0
e_1 = 0
e_2 = 0
i_2 = 0
o_2 = 0
c_2 = 125
w_2 = 0
# i_3 = 0
# o_3 = 0
# c_3 = 125
# w_3 = 0

for x in range(1, 300):
   #  client_socket.send(message.encode())
    message = '{} {} {} {} {} {} {} {} {} {} '.format(i, o, c, w, e_1, e_2, i_2, o_2, c_2, w_2)
    client_socket.send(message.encode())

    data = client_socket.recv(1024)
    time.sleep(1)

    i += 1
    o += 1
    c += 0.009
    if w < 40:
       w += 1
    e_1 += 1
    if e_1 == 6:
       e_1 = 0
    e_2 += 1
    if e_2 == 12:
       e_2 = 0
    i_2 += 1
    o_2 += 1
    c_2 += 0.009
    if w_2 < 40:
       w_2 += 1
   #  i_3 += 1
   #  o_3 += 1
   #  c_3 += 0.009
   #  if w_3 < 40:
   #     w_3 += 1
    # print('Received from the server :',repr(data.decode()))


client_socket.close()