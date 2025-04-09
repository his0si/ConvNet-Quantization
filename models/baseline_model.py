import torch
import torch.nn as nn
import torchvision
import os

class BaselineModel:
    """
    Baseline 모델 (ResNet50)을 관리하는 클래스
    """
    def __init__(self):
        # 원본 FP32 모델 생성
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
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # Baseline 모델 생성
    baseline = BaselineModel()
    model = baseline.get_model()
    
    # 모델 크기 확인
    size_mb = baseline.get_model_size()
    print(f"Baseline 모델 크기: {size_mb:.2f} MB")
    
    # 테스트 입력으로 추론 테스트
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 모델 추론
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    # 출력 확인
    print(f"출력 형태: {output.shape}")
    
    return model

if __name__ == "__main__":
    test_baseline_model()

