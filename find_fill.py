import cv2


import torch
import torchvision.transforms as transforms
import onnxruntime
import numpy as np
from PIL import Image


# 이미지 전처리를 위한 변환 정의
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ONNX 모델을 ONNX Runtime으로 로드
ort_session = onnxruntime.InferenceSession('model_ft.onnx')

# 클래스 이름 목록
class_names = ['Elmuco_Tab', 'Gelusam_Tab', 'Lopmin_Cap', 'Tylenol_Powder_160mg']



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
            global roi_img
            roi_img= roi




    image = Image.fromarray(roi_img)
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)  # 배치 차원 추가

    # 추론 수행
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # 클래스별 확률 계산
    output_probs = torch.softmax(torch.tensor(ort_outs[0]), dim=1)

    # 가장 높은 확률을 가진 클래스 출력
    class_index = torch.argmax(output_probs[0]).item()
    predicted_class = class_names[class_index]

    # 결과를 화면에 출력
    print(f'Predicted class: {predicted_class}')




    cv2.imshow('contourFrame', contour_frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
