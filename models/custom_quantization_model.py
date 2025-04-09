import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import torchvision
import os

class CustomQuantization:
    """
    Custom Dynamic Quantization 방식을 구현한 클래스
    """
    def __init__(self):
        # 커스텀 양자화 모델 생성 (ResNet50)
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.quantized_model = None
    
    def quantize(self, model=None):
        """
        동적 양자화를 수행하는 함수
        """
        if model is not None:
            self.model = model
            
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # 동적 양자화 적용 - 커스텀 설정으로 특정 레이어에 다른 설정 적용 가능
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
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
def test_custom_quantization():
    from utils.dataset_manager import DatasetManager
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # Custom Quantization 모델 생성 및 양자화
    custom_quant = CustomQuantization()
    
    # 양자화 전 모델 크기 확인
    fp32_size = custom_quant.get_model_size(custom_quant.model)
    print(f"FP32 모델 크기: {fp32_size:.2f} MB")
    
    # 양자화 수행
    print("커스텀 동적 양자화 수행 중...")
    quantized_model = custom_quant.quantize()
    
    # 양자화 후 모델 크기 확인
    int8_size = custom_quant.get_model_size(quantized_model)
    print(f"커스텀 동적 양자화 모델 크기: {int8_size:.2f} MB")
    print(f"압축률: {fp32_size / int8_size:.2f}x")
    
    # 테스트 입력으로 추론 테스트
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # FP32 모델 추론
    custom_quant.model.eval()
    with torch.no_grad():
        fp32_output = custom_quant.model(dummy_input)
    
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
    test_custom_quantization()
