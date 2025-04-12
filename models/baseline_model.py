import torch
import torch.nn as nn
import torchvision
import os
from tqdm import tqdm

class BaselineModel:
    """
    Baseline 모델 (ResNet50)을 관리하는 클래스
    """
    def __init__(self):
        # 원본 FP32 모델 생성
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model = self.model.cpu()  # CPU로 이동
    
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

    def evaluate(self, test_loader):
        """
        모델의 Top-1, Top-5 정확도를 평가하는 함수
        """
        self.model.eval()
        correct1 = 0
        correct5 = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.cpu()
                labels = labels.cpu()
                
                outputs = self.model(images)
                _, predicted = outputs.topk(5, 1, True, True)
                predicted = predicted.t()
                correct = predicted.eq(labels.view(1, -1).expand_as(predicted))
                
                correct1 += correct[:1].reshape(-1).float().sum(0).item()
                correct5 += correct[:5].reshape(-1).float().sum(0).item()
                total += labels.size(0)
        
        top1 = 100 * correct1 / total
        top5 = 100 * correct5 / total
        return top1, top5

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
    
    # 모델 정확도 평가
    top1, top5 = baseline.evaluate(test_loader)
    print(f"Baseline 모델 정확도:")
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")
    
    return model

if __name__ == "__main__":
    test_baseline_model()
