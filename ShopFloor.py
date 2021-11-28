import pymysql.cursors
import socket
import cv2

# Server address
HOST = '192.168.0.114'
PORT = 6666

# Database setting
conn = pymysql.connect(
    host = 'localhost',
    user = 'root',
    passwd = '********',
    db = 'shop_floor',
    charset='utf8'
)
cursor = conn.cursor()

# Resolution
WIDTH = 1920
HEIGHT = 1080

# start line Coordinate
start1 = (1713,385)
start2 = (1813,385)

# end line Coordinate
end1 = (915,800)
end2 = (915,900)

# Error zone Coordinate
error1 = (1235,505)
error2 = (1260,810)

# GUI client setting
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Boundingbox color
basketball_color = (97,237,124)
soccerball_color = (90, 160, 230)

# Draw start,end line
def draw_line(frame,first,second):
    cv2.line(frame,first,second,(0,255,0),3)

# Draw indicator on Monitoring
def indicator(frame, total_cycletime, basket_cycletime, soccer_cycletime, count, basketball_count, soccerball_count,
              in_count, basket_in, soccer_in, out_count, basket_out, soccer_out):
    cv2.putText(frame, "Cycletime: Total {} (sec)".format(total_cycletime), (20, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0, 255, 0), 2)
    cv2.putText(frame, "B: Cycletime: {} (sec)".format(basket_cycletime), (150, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0, 255, 0), 2)
    cv2.putText(frame, "S: Cycletime: {} (sec)".format(soccer_cycletime), (150, 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0, 255, 0), 2)
    cv2.putText(frame, "WIP: Total {} (ea)".format(count), (25, 210), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
    cv2.putText(frame, "B: {} (ea)".format(basketball_count), (147, 250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0, 255, 0), 2)
    cv2.putText(frame, "S: {} (ea)".format(soccerball_count), (147, 290), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0, 255, 0), 2)
    cv2.putText(frame, "IN:  Total {} (ea)".format(in_count), (25, 360), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),2)
    cv2.putText(frame, "B: {} (ea)".format(basket_in), (142, 400), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
    cv2.putText(frame, "S: {} (ea)".format(soccer_in), (142, 440), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
    cv2.putText(frame, "OUT: Total {} (ea)".format(out_count), (16, 510), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0, 255, 0), 2)
    cv2.putText(frame, "B: {} (ea)".format(basket_out), (150, 550), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
    cv2.putText(frame, "S: {} (ea)".format(soccer_out), (150, 590), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)




