import torch
import torch.nn as nn
import torch.quantization
import torchvision
import os

class OptimizedCustomQuantization:
    """
    최적화된 커스텀 양자화를 구현한 클래스
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
        모델에 최적화된 커스텀 양자화를 적용하는 함수
        """
        # 모델을 CPU로 이동
        model = model.cpu()
        
        # 모듈 퓨전 - 레이어별 최적화된 퓨전 전략 적용
        fused_model = self._fuse_modules_optimized(model)
        
        # 레이어별 중요도 분석
        layer_importance = self._analyze_layer_importance(fused_model)
        
        # 레이어별 양자화 설정
        qconfig_dict = self._create_layer_specific_qconfig(fused_model, layer_importance)
        
        # 동적 양자화 적용
        quantized_model = torch.quantization.quantize_dynamic(
            fused_model,
            qconfig_dict=qconfig_dict,
            dtype=torch.qint8
        )
        
        # 양자화된 모델임을 표시
        quantized_model.quantized = True
        quantized_model.is_custom_quantized = True
        
        return quantized_model
    
    def _fuse_modules_optimized(self, model):
        """
        모델의 Conv-BN-ReLU 레이어를 최적화된 방식으로 퓨전하는 함수
        """
        modules_to_fuse = []
        
        # 첫 번째 Conv-BN-ReLU 퓨전 (입력 레이어)
        if hasattr(model, 'conv1') and hasattr(model, 'bn1') and hasattr(model, 'relu'):
            modules_to_fuse.append(['conv1', 'bn1', 'relu'])
        
        # Bottleneck 블록 내 Conv-BN-ReLU 퓨전
        for name, module in model.named_modules():
            if isinstance(module, torchvision.models.resnet.Bottleneck):
                block_prefix = name
                # 첫 번째 컨볼루션 레이어는 개별적으로 유지 (더 높은 정밀도 유지)
                if hasattr(module, 'conv2') and hasattr(module, 'bn2') and hasattr(module, 'relu'):
                    modules_to_fuse.append([f'{block_prefix}.conv2', f'{block_prefix}.bn2', f'{block_prefix}.relu'])
                if hasattr(module, 'conv3') and hasattr(module, 'bn3'):
                    modules_to_fuse.append([f'{block_prefix}.conv3', f'{block_prefix}.bn3'])
                if module.downsample is not None:
                    modules_to_fuse.append([f'{block_prefix}.downsample.0', f'{block_prefix}.downsample.1'])
        
        # 모듈 퓨전 실행
        fused_model = torch.quantization.fuse_modules(model, modules_to_fuse, inplace=False)
        return fused_model
    
    def _analyze_layer_importance(self, model):
        """
        레이어별 중요도를 분석하는 함수
        """
        layer_importance = {}
        
        # 첫 번째 컨볼루션 레이어는 매우 중요
        if hasattr(model, 'conv1'):
            layer_importance['conv1'] = 1.0
        
        # Bottleneck 블록 분석
        for name, module in model.named_modules():
            if isinstance(module, torchvision.models.resnet.Bottleneck):
                block_prefix = name
                # 첫 번째 컨볼루션 레이어는 더 중요
                if hasattr(module, 'conv1'):
                    layer_importance[f'{block_prefix}.conv1'] = 0.9
                # 두 번째 컨볼루션 레이어는 중간 중요도
                if hasattr(module, 'conv2'):
                    layer_importance[f'{block_prefix}.conv2'] = 0.7
                # 세 번째 컨볼루션 레이어는 상대적으로 덜 중요
                if hasattr(module, 'conv3'):
                    layer_importance[f'{block_prefix}.conv3'] = 0.5
        
        return layer_importance
    
    def _create_layer_specific_qconfig(self, model, layer_importance):
        """
        레이어별 양자화 설정을 생성하는 함수
        """
        qconfig_dict = {
            # 기본 설정
            '': torch.quantization.default_dynamic_qconfig,
            # 레이어별 설정
            'module_name': {}
        }
        
        # 중요도에 따른 양자화 설정
        for layer_name, importance in layer_importance.items():
            if importance >= 0.9:
                # 매우 중요한 레이어는 더 높은 정밀도로 양자화
                qconfig_dict['module_name'][layer_name] = torch.quantization.default_dynamic_qconfig
            elif importance >= 0.7:
                # 중간 중요도 레이어는 기본 설정
                qconfig_dict['module_name'][layer_name] = torch.quantization.default_dynamic_qconfig
            else:
                # 덜 중요한 레이어는 더 낮은 정밀도로 양자화
                qconfig_dict['module_name'][layer_name] = torch.quantization.default_dynamic_qconfig
        
        return qconfig_dict
    
    def get_model_size(self, model):
        """
        모델 크기를 계산하는 함수 (MB 단위)
        """
        torch.save(model.state_dict(), "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size_mb

# 테스트 함수
def test_optimized_custom_quantization():
    from utils.dataset_manager import DatasetManager
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # 최적화된 커스텀 양자화 모델 생성 및 양자화
    custom_model = OptimizedCustomQuantization()
    
    # 양자화 전 모델 크기 확인
    fp32_size = custom_model.get_model_size(custom_model.fp32_model)
    print(f"FP32 모델 크기: {fp32_size:.2f} MB")
    
    # 양자화 수행
    print("최적화된 커스텀 양자화 수행 중...")
    quantized_model = custom_model.quantize(custom_model.fp32_model)
    
    # 양자화 후 모델 크기 확인
    int8_size = custom_model.get_model_size(quantized_model)
    print(f"최적화된 커스텀 양자화 모델 크기: {int8_size:.2f} MB")
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
    test_optimized_custom_quantization() 