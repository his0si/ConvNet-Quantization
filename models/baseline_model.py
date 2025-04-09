import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    """
    간단한 컨볼루션 신경망 모델 - FP32 기준 모델
    """
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        # 첫 번째 컨볼루션 레이어
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 두 번째 컨볼루션 레이어
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 세 번째 컨볼루션 레이어
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)  # 10개 클래스 분류 (CIFAR-10 기준)
    
    def forward(self, x):
        # 첫 번째 블록
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 두 번째 블록
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 세 번째 블록
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 완전 연결 레이어
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 모델 생성 및 테스트 함수
def create_model():
    model = SimpleConvNet()
    return model

def test_model():
    model = create_model()
    # 32x32 RGB 이미지 (CIFAR-10 형식) 배치 생성
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    
    # 모델 추론 테스트
    output = model(dummy_input)
    
    print(f"모델 입력 크기: {dummy_input.shape}")
    print(f"모델 출력 크기: {output.shape}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")
    
    # 모델 구조 출력
    print("\n모델 구조:")
    print(model)
    
    return model

if __name__ == "__main__":
    test_model()

