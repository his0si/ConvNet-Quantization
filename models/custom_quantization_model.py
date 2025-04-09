import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import os

class CustomQuantization(nn.Module):
    """
    Custom Quantization 방식을 적용한 모델
    - QuantStub와 DeQuantStub를 모델 내부에 삽입
    - FloatFunctional을 사용하여 연산자 처리
    - 특정 레이어에 대한 커스텀 처리 가능
    """
    def __init__(self, base_model=None):
        super(CustomQuantization, self).__init__()
        
        # 양자화 관련 스텁 초기화
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # FloatFunctional 연산자 초기화
        self.add = nn.quantized.FloatFunctional()
        self.cat = nn.quantized.FloatFunctional()
        self.mul = nn.quantized.FloatFunctional()
        
        # 기본 모델 로드 또는 생성
        if base_model is None:
            self.base_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.base_model = base_model
            
        # 모델을 평가 모드로 설정
        self.base_model.eval()
        
        # 양자화된 모델임을 표시
        self.quantized = True
        self.is_custom_quantized = True
        
    def forward(self, x):
        # 입력 양자화
        x = self.quant(x)
        
        # 기본 모델의 forward pass
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        # ResNet 블록들 처리
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        # 평균 풀링
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 완전 연결 레이어
        x = self.base_model.fc(x)
        
        # 출력 역양자화
        x = self.dequant(x)
        
        return x
    
    def quantize(self, model=None):
        """
        모델에 커스텀 양자화를 적용하는 함수
        """
        if model is not None:
            self.base_model = model
            
        # 모델을 평가 모드로 설정
        self.base_model.eval()
        
        # 양자화 설정
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 모델 준비
        self.prepare(qconfig)
        
        # 캘리브레이션 데이터로 모델 준비
        # 실제 사용 시에는 적절한 캘리브레이션 데이터셋을 사용해야 함
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            self(dummy_input)
        
        # 양자화 변환
        self.convert()
        
        return self
    
    def prepare(self, qconfig):
        """
        모델을 양자화 준비 상태로 변환
        """
        # 모델을 CPU로 이동
        self.cpu()
        
        # 양자화 설정 적용
        self.qconfig = qconfig
        
        # 모듈 퓨전
        torch.quantization.fuse_modules(self.base_model, 
                                      [['conv1', 'bn1', 'relu']], 
                                      inplace=True)
        
        # 양자화 준비
        torch.quantization.prepare(self, inplace=True)
        
    def convert(self):
        """
        모델을 양자화된 상태로 변환
        """
        torch.quantization.convert(self, inplace=True)
        
    def get_model_size(self):
        """
        모델 크기를 계산하는 함수 (MB 단위)
        """
        torch.save(self.state_dict(), "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size_mb
