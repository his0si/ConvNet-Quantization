import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from models.custom_quantization_model import CustomQuantization
import torchvision
import torch
from torchvision import transforms
import seaborn as sns

class ResultAnalyzer:
    """
    모델 성능 분석 및 시각화를 위한 클래스
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def analyze_and_plot(self, results):
        """
        결과를 분석하고 시각화하는 함수
        """
        plt.style.use('default')  # seaborn 대신 default 스타일 사용
        sns.set_theme()  # seaborn 테마 적용
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 모델 이름 목록
        models = list(results['accuracy'].keys())
        
        # 1. 정확도 그래프
        top1_acc = [results['accuracy'][m]['top1'] for m in models]
        top5_acc = [results['accuracy'][m]['top5'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, top1_acc, width, label='Top-1')
        axes[0].bar(x + width/2, top5_acc, width, label='Top-5')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].legend()
        
        # 2. 모델 크기 그래프
        sizes = [results['model_size'][m] for m in models]
        axes[1].bar(models, sizes)
        axes[1].set_ylabel('Model Size (MB)')
        axes[1].set_title('Model Size Comparison')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. 추론 속도 그래프
        speeds = [results['inference_speed'][m] for m in models]
        axes[2].bar(models, speeds)
        axes[2].set_ylabel('Throughput (images/sec)')
        axes[2].set_title('Inference Speed')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()

def test_analyzer():
    from utils.dataset_manager import DatasetManager
    import torchvision
    
    # 데이터셋 매니저 생성
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # 테스트용 모델 생성
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 분석기 생성 및 테스트
    analyzer = ResultAnalyzer()
    results = analyzer.compare_models({'ResNet50': model}, dataset_manager)
    
    return analyzer

if __name__ == "__main__":
    test_analyzer() 