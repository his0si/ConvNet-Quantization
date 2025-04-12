import torch
import torch.nn as nn
import torch.quantization
import torchvision
from torch.quantization import QuantStub, DeQuantStub, prepare, convert, fuse_modules, get_default_qconfig
import os

#############################################
# Helper: Safe Fusion 함수
#############################################
def safe_fuse(module, fuse_list):
    """
    module 내부의 fuse_list에 해당하는 서브 모듈들이 모두 fuse 가능한지 확인 후,
    fuse 가능한 경우에만 fuse_modules() 호출.
    fuse_list: 예를 들어 ['conv2', 'bn2', 'relu']
    """
    children = dict(module.named_children())
    # fuse 가능한 모듈들: Conv2d, BatchNorm2d, ReLU (혹은 ReLU6)
    fuse_types = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.ReLU6)
    for name in fuse_list:
        sub_module = children.get(name, None)
        if sub_module is None or not isinstance(sub_module, fuse_types):
            # 이미 fuse되어 Identity이거나, 없는 경우엔 skip
            return
    # 모두 fuse 가능하면 fuse
    fuse_modules(module, fuse_list, inplace=True)

#############################################
# Custom Quantization Modules
#############################################
class CustomQuantizedConv2d(nn.Module):
    def __init__(self, conv_layer, custom_scale=None):
        """
        conv_layer: 원래의 Conv2d (혹은 fused Conv 모듈)
        custom_scale: 입력 스케일링 (옵션)
        """
        super().__init__()
        self.custom_scale = custom_scale
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # conv_layer에서 nn.Conv2d 모듈 추출
        conv_module = None
        if hasattr(conv_layer, "weight"):
            conv_module = conv_layer
        elif hasattr(conv_layer, "conv"):
            conv_module = conv_layer.conv
        else:
            for m in conv_layer.modules():
                if isinstance(m, nn.Conv2d):
                    conv_module = m
                    break
        if conv_module is None:
            raise AttributeError("전달된 conv_layer에서 weight 속성을 찾을 수 없습니다.")

        conv_module.weight.data = conv_module.weight.data.cpu()
        if conv_module.bias is not None:
            conv_module.bias.data = conv_module.bias.data.cpu()
        self.conv = conv_layer

    def forward(self, x):
        if x.is_quantized:
            x = x.dequantize()
        if self.custom_scale is not None:
            x = x * self.custom_scale
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

class CustomQuantizedLinear(nn.Module):
    def __init__(self, linear_layer, custom_scale=None):
        super().__init__()
        self.linear = linear_layer
        self.custom_scale = custom_scale
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.linear.weight.data = self.linear.weight.data.cpu()
        if self.linear.bias is not None:
            self.linear.bias.data = self.linear.bias.data.cpu()

    def forward(self, x):
        if x.is_quantized:
            x = x.dequantize()
        if self.custom_scale is not None:
            x = x * self.custom_scale
        x = self.quant(x)
        x = self.linear(x)
        x = self.dequant(x)
        return x

class CustomQuantizedBottleneck(nn.Module):
    def __init__(self, bottleneck, conv1_scale=None):
        super().__init__()
        self.conv1 = CustomQuantizedConv2d(bottleneck.conv1, conv1_scale)
        self.bn1 = bottleneck.bn1.cpu()
        self.conv2 = CustomQuantizedConv2d(bottleneck.conv2)
        self.bn2 = bottleneck.bn2.cpu()
        self.conv3 = CustomQuantizedConv2d(bottleneck.conv3)
        self.bn3 = bottleneck.bn3.cpu()
        self.relu = bottleneck.relu
        self.downsample = bottleneck.downsample
        self.stride = bottleneck.stride

        if self.downsample is not None:
            # downsample의 첫 번째 레이어를 교체
            self.downsample[0] = CustomQuantizedConv2d(self.downsample[0])
            if len(self.downsample) > 1:
                self.downsample[1] = self.downsample[1].cpu()

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

        # float 도메인에서 덧셈 수행
        if out.is_quantized:
            out = out.dequantize()
        if identity.is_quantized:
            identity = identity.dequantize()
        out = out + identity
        out = self.relu(out)
        return out

class CustomQuantizedResNet50(nn.Module):
    def __init__(self, model, conv1_scale=1.0):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        model = model.cpu()

        self.conv1 = CustomQuantizedConv2d(model.conv1, conv1_scale)
        self.bn1 = model.bn1.cpu()
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = nn.Sequential(*[CustomQuantizedBottleneck(b, conv1_scale) for b in model.layer1])
        self.layer2 = nn.Sequential(*[CustomQuantizedBottleneck(b) for b in model.layer2])
        self.layer3 = nn.Sequential(*[CustomQuantizedBottleneck(b) for b in model.layer3])
        self.layer4 = nn.Sequential(*[CustomQuantizedBottleneck(b) for b in model.layer4])

        self.avgpool = model.avgpool
        self.fc = CustomQuantizedLinear(model.fc)

    def forward(self, x):
        if x.is_quantized:
            x = x.dequantize()
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
        x = self.dequant(x)
        return x

class CustomQuantization:
    """
    커스텀 양자화를 구현한 클래스
    - 각 레이어별 독립적인 양자화/역양자화
    - 커스텀 스케일링 지원
    - CPU 기반 추론 최적화
    """
    def __init__(self, model=None, conv1_scale=2.0):
        if model is None:
            self.fp32_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.fp32_model = model
        self.conv1_scale = conv1_scale
        self.quantized_model = None
        
        # 양자화 백엔드 설정
        if 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
        elif 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        else:
            raise RuntimeError("No supported quantization engine found")
    
    def quantize(self):
        """
        커스텀 양자화를 수행하는 함수
        1. 모델을 CPU로 이동
        2. 평가 모드로 설정
        3. 모듈 퓨전 수행
        4. 커스텀 양자화 모델 생성
        """
        self.fp32_model = self.fp32_model.cpu()
        self.fp32_model.eval()
        
        # 모듈 퓨전
        fuse_model(self.fp32_model)
        
        # 커스텀 양자화 모델 생성
        self.quantized_model = CustomQuantizedResNet50(
            self.fp32_model,
            conv1_scale=self.conv1_scale
        )
        
        # 양자화 모델 속성 설정
        self.quantized_model.is_quantized = True
        self.quantized_model.is_custom_quantized = True
        
        return self.quantized_model
    
    def get_model_size(self):
        """
        모델 크기를 계산하는 함수 (MB 단위)
        """
        if self.quantized_model is None:
            model_to_check = self.fp32_model
        else:
            model_to_check = self.quantized_model
            
        # 전체 모델을 저장하여 크기 측정
        torch.save(model_to_check.state_dict(), "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size_mb

#############################################
# Fuse 함수 (ResNet50 기준, safe_fuse 사용)
#############################################
def fuse_model(model):
    # conv1, bn1, relu가 이미 fusion된 경우 확인
    if not isinstance(model.conv1, torch.nn.Conv2d):
        print("conv1, bn1, relu가 이미 fusion되어 있습니다.")
    else:
        safe_fuse(model, ['conv1', 'bn1', 'relu'])
    
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for basic_block in layer:
            if not isinstance(basic_block.conv1, torch.nn.Conv2d):
                print(f"{layer_name}의 conv1, bn1, relu가 이미 fusion되어 있습니다.")
            else:
                safe_fuse(basic_block, ['conv1', 'bn1', 'relu'])
            
            if not isinstance(basic_block.conv2, torch.nn.Conv2d):
                print(f"{layer_name}의 conv2, bn2, relu가 이미 fusion되어 있습니다.")
            else:
                # conv2, bn2, relu를 fuse 시도, 만약 safe_fuse 내부에서 skip되면 아무 변화 없음
                safe_fuse(basic_block, ['conv2', 'bn2', 'relu'])
                # 만약 위가 skip되었다면 conv2, bn2만 fuse 시도
                safe_fuse(basic_block, ['conv2', 'bn2'])
            
            if not isinstance(basic_block.conv3, torch.nn.Conv2d):
                print(f"{layer_name}의 conv3, bn3가 이미 fusion되어 있습니다.")
            else:
                safe_fuse(basic_block, ['conv3', 'bn3'])
            
            if basic_block.downsample:
                if not isinstance(basic_block.downsample[0], torch.nn.Conv2d):
                    print(f"{layer_name}의 downsample이 이미 fusion되어 있습니다.")
                else:
                    safe_fuse(basic_block.downsample, ['0', '1'])

#############################################
# 테스트 함수
#############################################
def test_custom_quantization():
    from utils.dataset_manager import DatasetManager
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # 커스텀 양자화 모델 생성 및 양자화 수행
    custom_quant = CustomQuantization()
    quantized_model = custom_quant.quantize()
    
    # 모델 크기 확인
    fp32_size = custom_quant.get_model_size()
    print(f"커스텀 양자화 모델 크기: {fp32_size:.2f} MB")
    
    # 더미 입력을 통한 추론 테스트
    dummy_input = torch.randn(1, 3, 224, 224)
    quantized_model.eval()
    with torch.no_grad():
        output = quantized_model(dummy_input)
    
    print(f"출력 형태: {output.shape}")
    return quantized_model

if __name__ == "__main__":
    test_custom_quantization()
