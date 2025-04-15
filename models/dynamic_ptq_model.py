import torch
import torch.nn as nn
import torch.quantization
import torchvision
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch.nn.quantized import FloatFunctional
import os

#############################################
# Custom Dynamic Quantized Modules
#############################################

class CustomDynamicQuantizedConv2d(nn.Module):
    """
    Conv2d에 대해 내부에서 양자화를 적용하는 커스텀 모듈.
    입력에 대해 QuantStub, 출력에 대해 DeQuantStub를 사용하며,
    conv1의 경우 custom_scale 값을 곱해 별도 스케일 조정을 할 수 있음.
    """
    def __init__(self, conv_layer, custom_scale=None):
        super().__init__()
        self.custom_scale = custom_scale
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv = conv_layer  # 원래 Conv2d 모듈 사용
        
    def forward(self, x):
        x = self.quant(x)
        if self.custom_scale is not None:
            x = x * self.custom_scale
        x = self.conv(x)
        x = self.dequant(x)
        return x

class CustomDynamicQuantizedLinear(nn.Module):
    """
    Linear 레이어에 대해 내부 양자화 적용.
    """
    def __init__(self, linear_layer, custom_scale=None):
        super().__init__()
        self.custom_scale = custom_scale
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.linear = linear_layer
    def forward(self, x):
        x = self.quant(x)
        if self.custom_scale is not None:
            x = x * self.custom_scale
        x = self.linear(x)
        x = self.dequant(x)
        return x

class CustomDynamicQuantizedBottleneck(nn.Module):
    """
    ResNet Bottleneck block을 커스텀 동적 양자화 방식으로 재구성.
    내부에 양자화 스텁을 삽입하고, 잔차 연결(add)을 FloatFunctional로 감싸 안전하게 계산합니다.
    conv1에 대해서는 별도 custom_scale 적용 가능.
    """
    def __init__(self, bottleneck, conv1_scale=None):
        super().__init__()
        self.conv1 = CustomDynamicQuantizedConv2d(bottleneck.conv1, conv1_scale)
        self.bn1 = bottleneck.bn1  # BatchNorm은 그대로 사용
        self.conv2 = CustomDynamicQuantizedConv2d(bottleneck.conv2)
        self.bn2 = bottleneck.bn2
        self.conv3 = CustomDynamicQuantizedConv2d(bottleneck.conv3)
        self.bn3 = bottleneck.bn3
        self.relu = bottleneck.relu
        self.downsample = bottleneck.downsample  # 하위 연결이 있다면 그대로 사용
        self.add_func = FloatFunctional()  # quantized tensor의 덧셈에 사용
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # FloatFunctional을 사용해 덧셈 (양자화 상태가 달라도 안전하게 수행)
        out = self.add_func.add(out, identity)
        out = self.relu(out)
        return out

class CustomDynamicQuantizedResNet50(nn.Module):
    """
    ResNet50 전체를 Custom Dynamic Quantization 방식으로 재구성.
    네트워크 입력에 대해 QuantStub, 출력에 대해 DeQuantStub를 적용하며,
    각 bottleneck block은 custom 방식으로 구성합니다.
    """
    def __init__(self, model, conv1_scale=1.0):
        super().__init__()
        # 모델 전체 입력과 최종 출력을 internal 양자화로 처리
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # 첫번째 conv layer는 커스텀 스케일 적용 가능
        self.conv1 = CustomDynamicQuantizedConv2d(model.conv1, conv1_scale)
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        
        # 각 layer의 Bottleneck을 커스텀 동적 양자화 block으로 대체
        self.layer1 = nn.Sequential(*(CustomDynamicQuantizedBottleneck(b, conv1_scale) for b in model.layer1))
        self.layer2 = nn.Sequential(*(CustomDynamicQuantizedBottleneck(b) for b in model.layer2))
        self.layer3 = nn.Sequential(*(CustomDynamicQuantizedBottleneck(b) for b in model.layer3))
        self.layer4 = nn.Sequential(*(CustomDynamicQuantizedBottleneck(b) for b in model.layer4))
        
        self.avgpool = model.avgpool
        self.fc = CustomDynamicQuantizedLinear(model.fc)
        
    def forward(self, x):
        # 내부에서 입력 양자화 시작
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # 출력 역양자화
        x = self.dequant(x)
        return x

#############################################
# Custom Dynamic Quantization 클래스를 통한 전체 모델 구성
#############################################
class CustomDynamicQuantization:
    """
    학습된 ResNet50 모델을 기반으로 custom dynamic 양자화를 적용하는 클래스.
    
    1. Fusing: Conv+BN+(ReLU)를 먼저 fuse합니다.
    2. 내부 모듈을 custom quantized 모듈로 재구성합니다.
    3. 모델 내부에서 QuantStub/DeQuantStub와 FloatFunctional로 연산을 안전하게 처리합니다.
    """
    def __init__(self, model, conv1_scale=1.0):
        self.model = model
        self.quantized_model = None
        self.conv1_scale = conv1_scale

    def quantize(self):
        self.model = self.model.cpu()
        self.model.eval()
        # 먼저 fusing 수행: 양자화 전에 Conv, BN, ReLU를 합쳐줍니다.
        fuse_model(self.model)
        # Custom dynamic quantized 모델 생성
        self.quantized_model = CustomDynamicQuantizedResNet50(self.model, self.conv1_scale)
        return self.quantized_model

    def get_model_size(self, model):
        # 전체 모델을 파일로 저장해서 크기를 측정 (state_dict로 저장시 누락되는 내부 정보 방지)
        torch.save(model, "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size_mb

#############################################
# Fuse 함수 (ResNet50 기준)
#############################################
def fuse_model(model):
    fuse_modules(model, ['conv1', 'bn1', 'relu'], inplace=True)
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for basic_block in layer:
            fuse_modules(basic_block, ['conv1', 'bn1', 'relu'], inplace=True)
            try:
                fuse_modules(basic_block, ['conv2', 'bn2', 'relu'], inplace=True)
            except Exception:
                fuse_modules(basic_block, ['conv2', 'bn2'], inplace=True)
            fuse_modules(basic_block, ['conv3', 'bn3'], inplace=True)
            if basic_block.downsample:
                fuse_modules(basic_block.downsample, ['0', '1'], inplace=True)

#############################################
# 테스트 함수
#############################################
def test_custom_dynamic_quantization():
    # pretrained ResNet50 모델 로드 (ImageNet1K)
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    fuse_model(model)
    
    quantizer = CustomDynamicQuantization(model, conv1_scale=2.0)
    print("Custom Dynamic Quantization 모델 생성 및 양자화 중...")
    quantized_model = quantizer.quantize()
    
    original_size = quantizer.get_model_size(model)
    quantized_size = quantizer.get_model_size(quantized_model)
    print(f"원본 모델 크기: {original_size:.2f} MB")
    print(f"양자화된 모델 크기: {quantized_size:.2f} MB")
    print(f"압축률: {original_size / quantized_size:.2f}x")
    
    return quantized_model

def test_model_inference():
    quantized_model = test_custom_dynamic_quantization()
    dummy_input = torch.randn(1, 3, 224, 224)
    quantized_model.eval()
    with torch.no_grad():
        output = quantized_model(dummy_input)
    print("추론 결과 출력 형태:", output.shape)

class DynamicPTQModel:
    """
    동적 양자화(Dynamic Quantization)를 구현한 클래스
    """
    def __init__(self):
        from models.baseline_model import SimpleConvNet
        self.fp32_model = SimpleConvNet()
        self.quantized_model = None
        
        if 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
        elif 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        else:
            raise RuntimeError("No supported quantization engine found")
    
    def load_state_dict(self, state_dict):
        """
        학습된 가중치를 로드하는 함수
        """
        self.fp32_model.load_state_dict(state_dict)
    
    def eval(self):
        """
        평가 모드로 설정
        """
        if self.quantized_model is not None:
            self.quantized_model.eval()
        else:
            self.fp32_model.eval()
        return self
    
    def cpu(self):
        """
        모델을 CPU로 이동
        """
        if self.quantized_model is not None:
            self.quantized_model = self.quantized_model.cpu()
        else:
            self.fp32_model = self.fp32_model.cpu()
        return self
    
    def to(self, device):
        """
        모델을 지정된 디바이스로 이동
        """
        if self.quantized_model is not None:
            self.quantized_model = self.quantized_model.to(device)
        else:
            self.fp32_model = self.fp32_model.to(device)
        return self
    
    def forward(self, x):
        """
        순전파 함수
        """
        if self.quantized_model is not None:
            return self.quantized_model(x)
        return self.fp32_model(x)
    
    def __call__(self, x):
        return self.forward(x)
    
    def quantize(self):
        """
        동적 양자화를 수행하는 함수
        """
        self.fp32_model = self.fp32_model.cpu()
        self.fp32_model.eval()
        
        # 모듈 퓨전 수행 - SimpleConvNet의 구조에 맞게 수정
        self.fp32_model = torch.quantization.fuse_modules(
            self.fp32_model,
            [['conv1', 'bn1'],
             ['conv2', 'bn2'],
             ['conv3', 'bn3'],
             ['conv4', 'bn4'],
             ['conv5', 'bn5'],
             ['conv6', 'bn6'],
             ['fc1', 'bn7']],
            inplace=False
        )
        
        # 동적 양자화 수행
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.fp32_model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        return self.quantized_model
    
    def get_model_size(self):
        """
        모델 크기를 계산하는 함수 (MB 단위)
        """
        torch.save(self.quantized_model.state_dict(), "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size_mb

def test_dynamic_ptq():
    from utils.dataset_manager import DatasetManager
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # 동적 양자화 모델 생성 및 양자화 수행
    ptq_model = DynamicPTQModel()
    quantized_model = ptq_model.quantize()
    
    # 모델 크기 확인
    size_mb = ptq_model.get_model_size()
    print(f"동적 양자화 모델 크기: {size_mb:.2f} MB")
    
    # 더미 입력을 통한 추론 테스트
    dummy_input = torch.randn(1, 3, 224, 224)
    quantized_model.eval()
    with torch.no_grad():
        output = quantized_model(dummy_input)
    
    print(f"출력 형태: {output.shape}")
    return quantized_model

if __name__ == "__main__":
    test_dynamic_ptq()
