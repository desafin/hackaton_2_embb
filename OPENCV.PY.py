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
while True:
    ret, frame = cap.read()  # 웹캠으로부터 프레임 읽기
    if not ret:
        break

    # 프레임 전처리
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    image = Image.fromarray(frame)
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)  # 배치 차원 추가

    # 추론 수행
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # 클래스별 확률 계산
    output_probs = torch.softmax(torch.tensor(ort_outs[0]), dim=1)

    # 가장 높은 확률을 가진 클래스 정보 출력
    max_prob = torch.max(output_probs[0])
    max_prob_index = torch.argmax(output_probs[0])
    predicted_class = class_names[max_prob_index]

    # 클래스 이름과 해당 확률을 함께 표시
    for i, class_name in enumerate(class_names):
        prob = output_probs[0][i].item()
        text = f'{class_name}: {prob:.4f}'
        cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    # 가장 높은 확률의 클래스를 출력
    print(f'Predicted class: {predicted_class} ({max_prob:.4f})')


    cv2.imshow('Webcam Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
        break


cap.release()  # 웹캠 해제
cv2.destroyAllWindows()  # 창 닫기
