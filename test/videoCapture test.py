import cv2

# 영상 불러오기
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if cap.isOpened():
    # 만약 카메라가 실행되고 있다면,
    ret, a = cap.read()
    # a: 영상 프레임을 읽어오기

    while ret:
        ret, a = cap.read()
        cv2.imshow("camera", a)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        # 종료 커맨드(ESC)

cap.release()
cv2.destroyAllWindows()
