import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from models.custom_quantization_model import CustomQuantizationModel
import torchvision
import torch
from torchvision import transforms
import seaborn as sns
import pandas as pd

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

    def compare_quantization_methods(self, fp32_model, ptq_model, proposed_model, test_loader):
        """
        FP32, PTQ, 제안된 양자화 방식의 성능을 비교하는 함수
        
        Args:
            fp32_model: FP32 baseline 모델
            ptq_model: PTQ로 양자화된 모델
            proposed_model: 제안된 방식으로 양자화된 모델
            test_loader: 테스트 데이터 로더
        """
        results = {
            'Model': ['FP32', 'PTQ', 'Proposed'],
            'Top-1 Accuracy': [],
            'Top-5 Accuracy': [],
            'Model Size (MB)': [],
            'Throughput (images/sec)': [],
            'Memory Usage (MB)': [],
            'Latency (ms)': []
        }
        
        models = {
            'FP32': fp32_model,
            'PTQ': ptq_model,
            'Proposed': proposed_model
        }
        
        # 각 모델의 성능 측정
        for model_name, model in models.items():
            model = model.to(self.device)
            model.eval()
            
            # 정확도 측정
            correct_top1 = 0
            correct_top5 = 0
            total = 0
            total_time = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # 추론 시간 측정
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    outputs = model(images)
                    end_time.record()
                    
                    torch.cuda.synchronize()
                    total_time += start_time.elapsed_time(end_time)
                    
                    _, predicted = outputs.topk(5, 1, True, True)
                    total += labels.size(0)
                    correct_top1 += (predicted[:, 0] == labels).sum().item()
                    correct_top5 += (predicted == labels.view(-1, 1)).sum().item()
            
            # 결과 저장
            results['Top-1 Accuracy'].append(100 * correct_top1 / total)
            results['Top-5 Accuracy'].append(100 * correct_top5 / total)
            
            # 모델 크기 측정
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            results['Model Size (MB)'].append(model_size)
            
            # 처리량 계산 (이미지/초)
            throughput = len(test_loader.dataset) / (total_time / 1000)  # ms를 초로 변환
            results['Throughput (images/sec)'].append(throughput)
            
            # 메모리 사용량 측정
            memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)
            results['Memory Usage (MB)'].append(memory_usage)
            
            # 지연시간 계산 (ms/이미지)
            latency = total_time / len(test_loader.dataset)
            results['Latency (ms)'].append(latency)
        
        # 결과를 DataFrame으로 변환
        df = pd.DataFrame(results)
        
        # 결과 시각화
        self._plot_comparison_results(df)
        
        return df
    
    def _plot_comparison_results(self, df):
        """
        비교 결과를 시각화하는 함수
        """
        plt.style.use('default')
        sns.set_theme()
        
        # 2x2 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 정확도 비교
        x = np.arange(len(df['Model']))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, df['Top-1 Accuracy'], width, label='Top-1')
        axes[0, 0].bar(x + width/2, df['Top-5 Accuracy'], width, label='Top-5')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(df['Model'])
        axes[0, 0].legend()
        
        # 2. 모델 크기와 메모리 사용량
        axes[0, 1].bar(x - width/2, df['Model Size (MB)'], width, label='Model Size')
        axes[0, 1].bar(x + width/2, df['Memory Usage (MB)'], width, label='Memory Usage')
        axes[0, 1].set_ylabel('Size (MB)')
        axes[0, 1].set_title('Model Size and Memory Usage')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(df['Model'])
        axes[0, 1].legend()
        
        # 3. 처리량
        axes[1, 0].bar(df['Model'], df['Throughput (images/sec)'])
        axes[1, 0].set_ylabel('Throughput (images/sec)')
        axes[1, 0].set_title('Inference Throughput')
        
        # 4. 지연시간
        axes[1, 1].bar(df['Model'], df['Latency (ms)'])
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].set_title('Inference Latency')
        
        plt.tight_layout()
        plt.savefig('quantization_comparison.png')
        plt.close()
        
        # 결과를 CSV 파일로 저장
        df.to_csv('quantization_comparison.csv', index=False)

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