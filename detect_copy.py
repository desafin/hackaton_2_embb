import cv2

# 웹캠을 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미합니다.

while True:
    # 영상 프레임 읽어오기
    ret, frame = cap.read()

    # 프레임 읽기가 실패하면 종료
    if not ret:
        break

    # 프레임을 화면에 보여주기
    cv2.imshow('Webcam', frame)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)

    # 노이즈 제거를 위한 침식과 팽창 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded_frame = cv2.erode(binary_frame, kernel, iterations=1)
    dilated_frame = cv2.dilate(eroded_frame, kernel, iterations=1)

    cv2.imshow('Denoised Webcam', dilated_frame)

    contour_frame = frame.copy()

    # 컨투어 찾기
    contours, _ = cv2.findContours(dilated_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어에 바운딩 박스 그리기 및 이미지 잘라내어 실시간으로 imshow
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if 2000 < contour_area < 5000:  # 원하는 면적 범위 설정
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(contour_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 바운딩 박스 안의 이미지 잘라내기
            roi = frame[y:y + h, x:x + w]
            cv2.imshow('ROI', roi)

    cv2.imshow('contourFrame', contour_frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
