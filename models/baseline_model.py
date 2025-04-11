# baseline_dynamic.py
import torch
import torch.nn as nn
import torch.quantization
import torchvision
import os

class BaselineModel:
    """
    Baseline 모델 (ResNet50)을 관리하는 클래스
    """
    def __init__(self):
        # 원본 FP32 모델 생성 (ImageNet 가중치 적용)
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
    
    def get_model(self):
        """
        모델을 반환하는 함수
        """
        return self.model
    
    def get_model_size(self):
        """
        모델 크기를 계산하는 함수 (MB 단위)
        """
        torch.save(self.model.state_dict(), "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size_mb

# 테스트 함수
def test_baseline_model():
    from utils.dataset_manager import DatasetManager
    
    # 데이터셋 로드 (실제 ImageNet 데이터셋 로드하는 코드)
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # Baseline 모델 생성
    baseline = BaselineModel()
    model = baseline.get_model()
    
    # 모델 크기 확인
    size_mb = baseline.get_model_size()
    print(f"Baseline 모델 크기: {size_mb:.2f} MB")
    
    # 더미 입력을 통한 추론 테스트
    dummy_input = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"출력 형태: {output.shape}")
    return model

if __name__ == "__main__":
    test_baseline_model()
    
    
################################################################################
    
class DynamicPTQModel:
    """
    동적 양자화(Dynamic Quantization)를 구현한 클래스
    """
    def __init__(self):
        self.fp32_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.quantized_model = None
        
        if 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
        elif 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        else:
            raise RuntimeError("No supported quantization engine found")
    
    def quantize(self, model):
        model = model.cpu()
        model.eval()
        fused_model = self._fuse_modules(model)
        quantized_model = torch.quantization.quantize_dynamic(
            fused_model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        quantized_model.quantized = True
        quantized_model.is_custom_quantized = False
        return quantized_model
    
    def _fuse_modules(self, model):
        modules_to_fuse = []
        if hasattr(model, 'conv1') and hasattr(model, 'bn1'):
            modules_to_fuse.append(['conv1', 'bn1'])
        
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
        
        fused_model = torch.quantization.fuse_modules(model, modules_to_fuse, inplace=False)
        return fused_model
    
    def get_model_size(self, model):
        torch.save(model.state_dict(), "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size_mb

def test_dynamic_ptq():
    from utils.dataset_manager import DatasetManager
    
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    ptq_model = DynamicPTQModel()
    fp32_size = ptq_model.get_model_size(ptq_model.fp32_model)
    print(f"FP32 모델 크기: {fp32_size:.2f} MB")
    
    print("동적 양자화 수행 중...")
    quantized_model = ptq_model.quantize(ptq_model.fp32_model)
    
    int8_size = ptq_model.get_model_size(quantized_model)
    print(f"동적 양자화 모델 크기: {int8_size:.2f} MB")
    print(f"압축률: {fp32_size / int8_size:.2f}x")
    
    dummy_input = torch.randn(1, 3, 224, 224)
    ptq_model.fp32_model.eval()
    with torch.no_grad():
        fp32_output = ptq_model.fp32_model(dummy_input)
    with torch.no_grad():
        int8_output = quantized_model(dummy_input)
    
    print(f"FP32 출력 형태: {fp32_output.shape}")
    print(f"INT8 출력 형태: {int8_output.shape}")
    
    output_diff = torch.abs(fp32_output - int8_output).mean().item()
    print(f"출력 평균 절대 차이: {output_diff:.6f}")
    
    return quantized_model

if __name__ == "__main__":
    test_dynamic_ptq()
