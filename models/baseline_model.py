import torch
import torchvision
import os
import torchvision.transforms as transforms

class BaselineModel:
    """
    Baseline 모델 (ResNet50)을 관리하는 클래스 (FP32)
    """
    def __init__(self):
        # torchvision의 사전 학습된 ResNet50 (IMAGENET1K_V1)을 그대로 사용합니다.
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
    
    def get_model(self):
        return self.model
    
    def get_model_size(self):
        torch.save(self.model.state_dict(), "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size_mb

def test_baseline_model():
    from utils.dataset_manager import DatasetManager
    
    # ImageNet 표준 전처리 적용
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms.mean, 
                             std=torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms.std),
    ])
    dataset_manager = DatasetManager(transform=transform)
    test_loader = dataset_manager.get_imagenet_dataset()
    
    baseline = BaselineModel()
    model = baseline.get_model()
    size_mb = baseline.get_model_size()
    print(f"Baseline 모델 크기: {size_mb:.2f} MB")
    
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"출력 형태: {output.shape}")
    
    return model

if __name__ == "__main__":
    test_baseline_model()
