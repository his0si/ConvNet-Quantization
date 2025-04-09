import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import os
from models.baseline_model import SimpleConvNet

class CustomQuantizedConvNet(nn.Module):
    """
    Custom Quantization 방식을 적용한 컨볼루션 신경망 모델
    """
    def __init__(self):
        super(CustomQuantizedConvNet, self).__init__()
        
        # 첫 번째 컨볼루션 레이어 (특별 처리 가능)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 두 번째 컨볼루션 레이어
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 세 번째 컨볼루션 레이어
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # 첫 번째 블록
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 두 번째 블록
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 세 번째 블록
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 완전 연결 레이어
        x = x.reshape(-1, 64 * 4 * 4)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def load_from_fp32(self, fp32_model):
        """
        FP32 모델의 가중치를 로드하는 함수
        """
        # 상태 사전 복사
        self.load_state_dict(fp32_model.state_dict())
        return self

class CustomDynamicQuantization:
    """
    커스텀 동적 양자화 방식을 구현한 클래스
    """
    def __init__(self, model_path=None):
        # 원본 FP32 모델 생성 또는 로드
        if model_path and os.path.exists(model_path):
            self.fp32_model = SimpleConvNet()
            self.fp32_model.load_state_dict(torch.load(model_path))
            print(f"학습된 모델을 {model_path}에서 로드했습니다.")
        else:
            self.fp32_model = SimpleConvNet()
            print("새 모델을 생성했습니다. (학습되지 않음)")
        
        # 커스텀 모델 생성 및 가중치 복사
        self.custom_model = CustomQuantizedConvNet()
        self.custom_model.load_from_fp32(self.fp32_model)
        
        # 양자화를 위한 모델 준비
        self.quantized_model = None
    
    def quantize(self):
        """
        커스텀 동적 양자화를 수행하는 함수
        """
        # 모델을 평가 모드로 설정
        self.custom_model.eval()
        
        # 동적 양자화 적용 - 커스텀 설정으로 특정 레이어에 다른 설정 적용
        qconfig_dict = {
            # 기본 설정
            '': torch.quantization.default_dynamic_qconfig,
            # 첫 번째 컨볼루션 레이어에 대한 특별 설정
            'module_name': {
                'conv1': torch.quantization.default_dynamic_qconfig
            }
        }
        
        # 동적 양자화 적용
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.custom_model,
            #qconfig_dict=qconfig_dict,  # 커스텀 설정 적용
            dtype=torch.qint8
        )
        
        return self.quantized_model
    
    def get_model_size(self, model):
        """
        모델 크기를 계산하는 함수 (MB 단위)
        """
        torch.save(model.state_dict(), "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size_mb

# 테스트 함수
def test_custom_dynamic_quantization(model_path=None):
    import torch.utils.data
    from torchvision import datasets, transforms
    from models.baseline_model import SimpleConvNet
    
    # 간단한 테스트 데이터셋 생성 (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 테스트용 데이터셋
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 커스텀 동적 양자화 모델 생성 및 양자화
    custom_quant = CustomDynamicQuantization(model_path)
    
    # 양자화 전 모델 크기 확인
    fp32_size = custom_quant.get_model_size(custom_quant.fp32_model)
    print(f"FP32 모델 크기: {fp32_size:.2f} MB")
    
    # 양자화 수행
    print("커스텀 동적 양자화 수행 중...")
    quantized_model = custom_quant.quantize()
    
    # 양자화 후 모델 크기 확인
    int8_size = custom_quant.get_model_size(quantized_model)
    print(f"커스텀 동적 양자화 모델 크기: {int8_size:.2f} MB")
    print(f"압축률: {fp32_size / int8_size:.2f}x")
    
    # 테스트 입력으로 추론 테스트
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # FP32 모델 추론
    custom_quant.fp32_model.eval()
    with torch.no_grad():
        fp32_output = custom_quant.fp32_model(dummy_input)
    
    # INT8 모델 추론
    with torch.no_grad():
        int8_output = quantized_model(dummy_input)
    
    # 출력 비교
    print(f"FP32 출력 형태: {fp32_output.shape}")
    print(f"INT8 출력 형태: {int8_output.shape}")
    
    # 출력 차이 계산
    output_diff = torch.abs(fp32_output - int8_output).mean().item()
    print(f"출력 평균 절대 차이: {output_diff:.6f}")
    
    # 정확도 평가
    from utils.model_evaluator import ModelEvaluator
    evaluator = ModelEvaluator(test_loader)
    
    print("\nFP32 모델 정확도 평가:")
    fp32_accuracy = evaluator.evaluate_accuracy(custom_quant.fp32_model)
    
    print("\n커스텀 동적 양자화 모델 정확도 평가:")
    int8_accuracy = evaluator.evaluate_accuracy(quantized_model)
    
    print(f"\n정확도 비교:")
    print(f"FP32 모델: {fp32_accuracy:.2f}%")
    print(f"커스텀 동적 양자화 모델: {int8_accuracy:.2f}%")
    print(f"정확도 차이: {fp32_accuracy - int8_accuracy:.2f}%")
    
    return quantized_model, fp32_accuracy, int8_accuracy

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_custom_dynamic_quantization(model_path)
