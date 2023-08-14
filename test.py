import torch
import torchvision.transforms as transforms
import onnx
import onnxruntime
from PIL import Image

# 이미지 전처리를 위한 변환 정의
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ONNX 모델 로드
model = onnx.load('model_ft.onnx')

# ONNX 모델을 ONNX Runtime으로 로드
ort_session = onnxruntime.InferenceSession('model_ft.onnx')

# 분류할 이미지 로드 및 전처리
image_path = 'K-004799_0_0_0_0_60_000_200.png'
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)  # 배치 차원 추가

# 추론 수행
ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
ort_outs = ort_session.run(None, ort_inputs)

# 클래스별 확률 계산 및 출력
output_probs = torch.softmax(torch.tensor(ort_outs[0]), dim=1)
class_names = ['Elmuco_Tab', 'Gelusam_Tab', 'Lopmin_Cap','Tylenol_Powder_160mg']  # 클래스 이름 목록 (필요에 따라 수정)

# 각 클래스의 확률 출력
for i, prob in enumerate(output_probs[0]):
    class_name = class_names[i]
    print(f'Class: {class_name}, Probability: {prob.item():.4f}')

# 가장 높은 확률을 가진 클래스 출력
class_index = torch.argmax(output_probs[0]).item()
predicted_class = class_names[class_index]
print(f'Predicted class: {predicted_class}')
