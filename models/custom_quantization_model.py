import torch
import torch.nn as nn
import torch.quantization
import torchvision
import os

class CustomQuantization:
    """
    커스텀 양자화를 구현한 클래스
    """
    def __init__(self):
        # 원본 FP32 모델 생성 (ResNet50)
        self.fp32_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.quantized_model = None
        
        # 양자화 백엔드 설정
        if 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
        elif 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        else:
            raise RuntimeError("No supported quantization engine found")
    
    def quantize(self, model):
        """
        모델에 커스텀 양자화를 적용하는 함수
        """
        # 모델을 CPU로 이동하고 평가 모드로 설정
        model = model.cpu()
        model.eval()  # 평가 모드로 설정
        
        # 모듈 퓨전 - 레이어별 최적화된 퓨전 전략 적용
        fused_model = self._fuse_modules_optimized(model)
        
        # 동적 양자화 적용
        quantized_model = torch.quantization.quantize_dynamic(
            fused_model,
            {nn.Linear, nn.Conv2d},  # 양자화할 레이어 타입
            dtype=torch.qint8  # 양자화 데이터 타입
        )
        
        # 양자화된 모델임을 표시
        quantized_model.quantized = True
        quantized_model.is_custom_quantized = True
        
        return quantized_model
    
    def _fuse_modules_optimized(self, model):
        """
        모델의 Conv-BN 레이어를 최적화된 방식으로 퓨전하는 함수
        """
        modules_to_fuse = []
        
        # 첫 번째 Conv-BN 퓨전
        if hasattr(model, 'conv1') and hasattr(model, 'bn1'):
            modules_to_fuse.append(['conv1', 'bn1'])
        
        # Bottleneck 블록 내 Conv-BN 퓨전
        for name, module in model.named_modules():
            if isinstance(module, torchvision.models.resnet.Bottleneck):
                block_prefix = name
                if hasattr(module, 'conv1') and hasattr(module, 'bn1'):
                    modules_to_fuse.append([f'{block_prefix}.conv1', f'{block_prefix}.bn1'])
                if hasattr(module, 'conv2') and hasattr(module, 'bn2'):
                    modules_to_fuse.append([f'{block_prefix}.conv2', f'{block_prefix}.bn2'])
                if hasattr(module, 'conv3') and hasattr(module, 'bn3'):
                    modules_to_fuse.append([f'{block_prefix}.conv3', f'{block_prefix}.bn3'])
                if module.downsample is not None:
                    modules_to_fuse.append([f'{block_prefix}.downsample.0', f'{block_prefix}.downsample.1'])
        
        # 모듈 퓨전 실행
        fused_model = torch.quantization.fuse_modules(model, modules_to_fuse, inplace=False)
        return fused_model
    
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
    
    # 커스텀 양자화 모델 생성 및 양자화
    custom_model = CustomQuantization()
    
    # 양자화 전 모델 크기 확인
    fp32_size = custom_model.get_model_size(custom_model.fp32_model)
    print(f"FP32 모델 크기: {fp32_size:.2f} MB")
    
    # 양자화 수행
    print("커스텀 양자화 수행 중...")
    quantized_model = custom_model.quantize(custom_model.fp32_model)
    
    # 양자화 후 모델 크기 확인
    int8_size = custom_model.get_model_size(quantized_model)
    print(f"커스텀 양자화 모델 크기: {int8_size:.2f} MB")
    print(f"압축률: {fp32_size / int8_size:.2f}x")
    
    # 테스트 입력으로 추론 테스트
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # FP32 모델 추론
    custom_model.fp32_model.eval()
    with torch.no_grad():
        fp32_output = custom_model.fp32_model(dummy_input)
    
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
