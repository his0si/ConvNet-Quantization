import torch
import torch.nn as nn
import torch.quantization
import torchvision
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional
import os

class OptimizedQuantizedConv2d(nn.Module):
    """
    최적화된 양자화된 컨볼루션 레이어
    """
    def __init__(self, conv):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        
        # 양자화된 연산을 위한 추가 최적화
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = nn.functional.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )
        return self.dequant(x)

class OptimizedQuantizedLinear(nn.Module):
    """
    최적화된 양자화된 선형 레이어
    """
    def __init__(self, linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        
        # 양자화된 연산을 위한 추가 최적화
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = nn.functional.linear(x, self.weight, self.bias)
        return self.dequant(x)

class CustomQuantizedResNet50(nn.Module):
    """
    최적화된 커스텀 양자화를 적용한 ResNet50 모델
    """
    def __init__(self, fp32_model):
        super(CustomQuantizedResNet50, self).__init__()
        
        # 양자화/역양자화 스텁 추가
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # 원본 모델의 모든 레이어 복사 및 최적화
        self.model = self._optimize_model(fp32_model)
        
        # 레이어별 양자화 설정
        self._setup_qconfig()
        
        # 양자화 준비
        self.prepare()
    
    def _optimize_model(self, model):
        """
        모델의 레이어를 최적화된 양자화 버전으로 변환
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(model, name, OptimizedQuantizedConv2d(module))
            elif isinstance(module, nn.Linear):
                setattr(model, name, OptimizedQuantizedLinear(module))
            else:
                self._optimize_model(module)
        return model
    
    def _setup_qconfig(self):
        """
        레이어별 양자화 설정 구성
        """
        # 기본 양자화 설정
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 레이어별 양자화 설정
        for name, module in self.model.named_modules():
            if isinstance(module, (OptimizedQuantizedConv2d, OptimizedQuantizedLinear)):
                module.qconfig = self.qconfig
    
    def prepare(self):
        """
        양자화를 위한 모델 준비
        """
        torch.quantization.prepare(self, inplace=True)
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)
    
    def quantize(self):
        """
        모델 양자화 실행
        """
        torch.quantization.convert(self, inplace=True)
        self.quantized = True
        self.is_custom_quantized = True
        return self

class CustomQuantization:
    """
    최적화된 커스텀 양자화를 관리하는 클래스
    """
    def __init__(self, model=None):
        # 원본 FP32 모델 생성 또는 로드
        if model is None:
            self.fp32_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.fp32_model = model
        
        # 모델을 CPU로 이동
        self.fp32_model = self.fp32_model.cpu()
        
        # 양자화 백엔드 설정
        if 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
        elif 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        else:
            raise RuntimeError("No supported quantization engine found")
    
    def quantize(self):
        """
        최적화된 커스텀 양자화를 수행하는 함수
        """
        # 모델을 평가 모드로 설정
        self.fp32_model.eval()
        
        # 최적화된 양자화 모델 생성
        quantized_model = CustomQuantizedResNet50(self.fp32_model)
        
        # 양자화 실행
        quantized_model.quantize()
        
        return quantized_model
    
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
    import os
    from utils.dataset_manager import DatasetManager
    from utils.model_evaluator import ModelEvaluator
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # 커스텀 양자화 모델 생성 및 양자화
    custom_quant = CustomQuantization()
    
    # 양자화 전 모델 크기 확인
    fp32_size = custom_quant.get_model_size(custom_quant.fp32_model)
    print(f"FP32 모델 크기: {fp32_size:.2f} MB")
    
    # 양자화 수행
    print("최적화된 커스텀 양자화 수행 중...")
    quantized_model = custom_quant.quantize()
    
    # 양자화 후 모델 크기 확인
    int8_size = custom_quant.get_model_size(quantized_model)
    print(f"최적화된 커스텀 양자화 모델 크기: {int8_size:.2f} MB")
    print(f"압축률: {fp32_size / int8_size:.2f}x")
    
    # 테스트 입력으로 추론 테스트
    dummy_input = torch.randn(1, 3, 224, 224)
    
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
    evaluator = ModelEvaluator(test_loader)
    
    print("\nFP32 모델 정확도 평가:")
    fp32_accuracy = evaluator.evaluate_accuracy(custom_quant.fp32_model)
    
    print("\n최적화된 커스텀 양자화 모델 정확도 평가:")
    int8_accuracy = evaluator.evaluate_accuracy(quantized_model)
    
    print(f"\n정확도 비교:")
    print(f"FP32 모델: {fp32_accuracy:.2f}%")
    print(f"최적화된 커스텀 양자화 모델: {int8_accuracy:.2f}%")
    print(f"정확도 차이: {fp32_accuracy - int8_accuracy:.2f}%")
    
    return quantized_model, fp32_accuracy, int8_accuracy

if __name__ == "__main__":
    test_custom_quantization()
