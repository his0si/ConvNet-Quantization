import torch
import torch.nn as nn
import torchvision
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert, fuse_modules
from torch.quantization import get_default_qat_qconfig
import os

# --- Custom Quantization Modules (서브모듈에서는 quantization stub 제거) ---

class CustomQuantizedConv2d(nn.Module):
    def __init__(self, conv_layer, custom_scale=None):
        """
        conv_layer: 일반 Conv2d 또는 fused ConvReLU2d 등
        custom_scale: 입력 scaling 적용 (옵션)
        """
        super().__init__()
        self.custom_scale = custom_scale
        
        # 전달받은 conv_layer에서 실제 nn.Conv2d 모듈을 찾아 CPU로 이동
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
        # 내부 quantization stub 제거 → global quantization (최상위 QuantStub 사용)
        x = self.conv(x)
        return x

class CustomQuantizedLinear(nn.Module):
    def __init__(self, linear_layer, custom_scale=None):
        super().__init__()
        self.custom_scale = custom_scale
        self.linear = linear_layer
        self.linear.weight.data = self.linear.weight.data.cpu()
        if self.linear.bias is not None:
            self.linear.bias.data = self.linear.bias.data.cpu()

    def forward(self, x):
        if x.is_quantized:
            x = x.dequantize()
        if self.custom_scale is not None:
            x = x * self.custom_scale
        x = self.linear(x)
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
        
        # 안전한 residual 덧셈: 양자화된 텐서를 dequantize한 후 float 덧셈 실행
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
        # 최상위에서 global QuantStub/DeQuantStub 사용
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
    def __init__(self, model, conv1_scale=1.0):
        self.model = model
        self.quantized_model = None
        self.conv1_scale = conv1_scale

    def quantize(self, calibration_loader=None):
        self.model = self.model.cpu()
        self.model.eval()
        fuse_model(self.model)
        self.quantized_model = CustomQuantizedResNet50(self.model, self.conv1_scale)
        self.quantized_model.qconfig = get_default_qat_qconfig('fbgemm')

        # QAT 준비
        self.quantized_model = prepare_qat(self.quantized_model)
        
        # Calibration 단계: dummy forward pass 5회로 observer 업데이트
        self.quantized_model.train()
        with torch.no_grad():
            for _ in range(5):
                dummy = torch.randn(1, 3, 224, 224)
                self.quantized_model(dummy)
        self.quantized_model.eval()
        
        # 변환 수행
        self.quantized_model = convert(self.quantized_model)
        self.quantized_model = self.quantized_model.cpu()
        return self.quantized_model

    def get_model_size(self, model):
        param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
        return (param_size + buffer_size) / 1024**2

# --- Fusion 함수 (ResNet50 기준) ---
def fuse_model(model):
    fuse_modules(model, ['conv1', 'bn1', 'relu'], inplace=True)
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for basic_block in layer:
            fuse_modules(basic_block, ['conv1', 'bn1', 'relu'], inplace=True)
            try:
                fuse_modules(basic_block, ['conv2', 'bn2', 'relu'], inplace=True)
            except AssertionError:
                fuse_modules(basic_block, ['conv2', 'bn2'], inplace=True)
            fuse_modules(basic_block, ['conv3', 'bn3'], inplace=True)
            if basic_block.downsample:
                fuse_modules(basic_block.downsample, ['0', '1'], inplace=True)

def test_custom_quantization():
    model = torchvision.models.resnet50(pretrained=True)
    fuse_model(model)

    quantizer = CustomQuantization(model, conv1_scale=2.0)
    quantized_model = quantizer.quantize(calibration_loader=None)

    original_size = quantizer.get_model_size(model)
    quantized_size = quantizer.get_model_size(quantized_model)

    print(f"원본 모델 크기: {original_size:.2f} MB")
    print(f"양자화된 모델 크기: {quantized_size:.2f} MB")
    print(f"압축률: {original_size/quantized_size:.2f}x")
    return quantized_model

if __name__ == "__main__":
    test_custom_quantization()
