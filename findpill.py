import torch
import torchvision.transforms as transforms
import onnxruntime
import cv2
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

# 웹캠에서 실시간 영상 캡처 및 분류
cap = cv2.VideoCapture(0)  # 웹캠 캡처 객체 생성

def classify_frame(frame):
    # 프레임 전처리
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    image = Image.fromarray(frame)
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

    print(predicted_class)
    return predicted_class

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(f"Using device: {device}")
ort_session.set_providers([f'CUDAExecutionProvider'])
ort_session.set_providers([f'CPUExecutionProvider'])

while True:
    ret, frame = cap.read()  # 웹캠으로부터 프레임 읽기
    if not ret:
        break

    # 각 약의 분류 결과 출력
    for i, x1, y1, x2, y2 in [[0, 0, 0, 100, 100], [1, 100, 0, 200, 100], [2, 200, 0, 300, 100], [3, 300, 0, 400, 100]]:
        sub_frame = frame[y1:y2, x1:x2]
        predicted_class = classify_frame(sub_frame)
        cv2.putText(frame, f'Class {i}: {predicted_class}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 결과를 화면에 출력
    cv2.imshow('Webcam Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
        break

cap.release()  # 웹캠 해제
cv2.destroyAllWindows()  # 창 닫기
