import torch
import torch.nn as nn
import torch.quantization
import torchvision
from torch.quantization import QuantStub, DeQuantStub, prepare, convert, fuse_modules, get_default_qconfig
import os
from models.baseline_model import SimpleConvNet
from torch.nn.quantized import FloatFunctional
import torch.nn.functional as F

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
    def __init__(self, conv_layer):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv = conv_layer

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

class CustomQuantizedLinear(nn.Module):
    def __init__(self, linear_layer):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.linear = linear_layer

    def forward(self, x):
        x = self.quant(x)
        x = self.linear(x)
        x = self.dequant(x)
        return x

class CustomQuantizedBottleneck(nn.Module):
    def __init__(self, bottleneck, conv1_scale=None):
        super().__init__()
        self.conv1 = CustomQuantizedConv2d(bottleneck.conv1)
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

        self.conv1 = CustomQuantizedConv2d(model.conv1)
        self.bn1 = model.bn1.cpu()
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = nn.Sequential(*[CustomQuantizedBottleneck(b) for b in model.layer1])
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

class CustomQuantizationModel(nn.Module):
    """
    커스텀 양자화를 적용한 모델
    """
    def __init__(self):
        super(CustomQuantizationModel, self).__init__()
        self.model = SimpleConvNet()
        self.quantized_model = None
        self.is_custom_quantized = True
        
        # 양자화 엔진 설정
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
        self.model.load_state_dict(state_dict)
    
    def quantize(self):
        """
        모델을 커스텀 양자화하는 함수
        """
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # 모델을 CPU로 이동
        self.model = self.model.cpu()
        
        # 모듈 퓨전 수행 - SimpleConvNet의 실제 구조에 맞게 수정
        self.model = torch.quantization.fuse_modules(
            self.model,
            [['conv1', 'bn1'],
             ['conv2', 'bn2'],
             ['conv3', 'bn3'],
             ['conv4', 'bn4'],
             ['conv5', 'bn5'],
             ['conv6', 'bn6'],
             ['fc1', 'bn7']],
            inplace=False
        )
        
        # 커스텀 양자화 모델 생성
        self.quantized_model = CustomQuantizedSimpleConvNet(self.model)
        
        return self.quantized_model
    
    def forward(self, x):
        if self.quantized_model is not None:
            return self.quantized_model(x)
        return self.model(x)

class CustomQuantizedSimpleConvNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.is_custom_quantized = True
        
        # 컨볼루션 레이어
        self.conv1 = CustomQuantizedConv2d(model.conv1)
        self.conv2 = CustomQuantizedConv2d(model.conv2)
        self.conv3 = CustomQuantizedConv2d(model.conv3)
        self.conv4 = CustomQuantizedConv2d(model.conv4)
        self.conv5 = CustomQuantizedConv2d(model.conv5)
        self.conv6 = CustomQuantizedConv2d(model.conv6)
        
        # 완전연결 레이어
        self.fc1 = CustomQuantizedLinear(model.fc1)
        self.fc2 = model.fc2  # 마지막 레이어는 양자화하지 않음
        
        # 풀링과 드롭아웃
        self.pool1 = model.pool1
        self.pool2 = model.pool2
        self.pool3 = model.pool3
        self.dropout1 = model.dropout1
        self.dropout2 = model.dropout2
        self.dropout3 = model.dropout3
        self.dropout4 = model.dropout4
        
        # FloatFunctional for safe operations
        self.add_func = FloatFunctional()
        
    def forward(self, x):
        x = self.quant(x)
        
        # 첫 번째 블록
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 두 번째 블록
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 세 번째 블록
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 완전연결 레이어
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        x = self.dequant(x)
        return x

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
    # 모델 테스트
    model = CustomQuantizationModel()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    return model

if __name__ == "__main__":
    test_custom_quantization()
