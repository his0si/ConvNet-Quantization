import torch
import torch.nn as nn
import torch.quantization
import os
from models.baseline_model import SimpleConvNet

class StaticPTQModel:
    """
    동적 양자화(Dynamic Quantization)를 구현한 클래스
    """
    def __init__(self):
        # 원본 FP32 모델 생성
        self.fp32_model = SimpleConvNet()
        
        
        # 양자화를 위한 모델 준비
        self.quantized_model = None
    
    def quantize(self, calibration_data_loader=None):
        """
        동적 양자화를 수행하는 함수
        """
        # 모델을 평가 모드로 설정
        self.fp32_model.eval()
        
        # 동적 양자화 적용
        # 가중치는 정적으로 양자화하고, 활성화는 동적으로 양자화
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.fp32_model,  # 양자화할 모델
            {nn.Linear, nn.Conv2d},  # 양자화할 레이어 타입
            dtype=torch.qint8  # 양자화 데이터 타입
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
def test_static_ptq():
    import os
    import torch.utils.data
    from torchvision import datasets, transforms
    
    # 간단한 테스트 데이터셋 생성 (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 테스트용 데이터셋 (일부만 사용)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # PTQ 모델 생성 및 양자화
    ptq_model = StaticPTQModel()
    
    # 양자화 전 모델 크기 확인
    fp32_size = ptq_model.get_model_size(ptq_model.fp32_model)
    print(f"FP32 모델 크기: {fp32_size:.2f} MB")
    
    # 양자화 수행
    print("동적 양자화 수행 중...")
    quantized_model = ptq_model.quantize()
    
    # 양자화 후 모델 크기 확인
    int8_size = ptq_model.get_model_size(quantized_model)
    print(f"동적 양자화 모델 크기: {int8_size:.2f} MB")
    print(f"압축률: {fp32_size / int8_size:.2f}x")
    
    # 테스트 입력으로 추론 테스트
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # FP32 모델 추론
    ptq_model.fp32_model.eval()
    with torch.no_grad():
        fp32_output = ptq_model.fp32_model(dummy_input)
    
    # INT8 모델 추론
    with torch.no_grad():
        int8_output = quantized_model(dummy_input)
    
    # 출력 비교
    print(f"FP32 출력 형태: {fp32_output.shape}")
    print(f"INT8 출력 형태: {int8_output.shape}")
    
    # 출력 차이 계산
    output_diff = torch.abs(fp32_output - int8_output).mean().item()
    print(f"출력 평균 절대 차이: {output_diff:.6f}")
    
    return quantized_model

if __name__ == "__main__":
    import os
    test_static_ptq()
