import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from models.custom_quantization_model import CustomQuantizationModel
import torchvision
<<<<<<< Updated upstream
=======
import torch
from torchvision import transforms
import seaborn as sns
import pandas as pd
>>>>>>> Stashed changes

class ResultAnalyzer:
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # Load a base model for quantization
        base_model = torchvision.models.resnet50(pretrained=True)
        self.quantization = CustomQuantization(base_model)
        
    def analyze_results(self, models_dict, speed_results):
        """
        모델 결과를 분석하고 시각화하는 메서드
        
        Args:
            models_dict (dict): 모델 정보를 담은 딕셔너리
            speed_results (dict): 속도 측정 결과를 담은 딕셔너리
        """
        # 결과 저장을 위한 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.save_dir, timestamp)
        os.makedirs(result_dir, exist_ok=True)
        
        # 모델 크기 정보 추가
        models_with_size = {}
        for name, model in models_dict.items():
            if name == 'Custom Quantization':
                # Custom Quantization 모델의 경우 양자화된 모델을 사용
                quantized_model = self.quantization.quantize()
                size_mb = self.quantization.get_model_size(quantized_model)
            else:
                size_mb = self.quantization.get_model_size(model)
            models_with_size[name] = {
                'model': model,
                'size': size_mb
            }
        
        # 모델 크기 비교
        self._plot_model_sizes(models_with_size, result_dir)
        
        # 처리 속도 비교
        self._plot_processing_speed(speed_results, result_dir)
        
        # 결과 요약 파일 생성
        self._generate_summary(models_with_size, speed_results, result_dir)
        
    def _plot_model_sizes(self, models_dict, result_dir):
        """모델 크기 비교 그래프 생성"""
        model_names = []
        sizes = []
        
        for name, model_info in models_dict.items():
            model_names.append(name)
            sizes.append(model_info['size'])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, sizes)
        plt.title('Model Sizes Comparison')
        plt.xlabel('Model')
        plt.ylabel('Size (MB)')
        plt.xticks(rotation=45)
        
        # 막대 위에 크기 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}MB',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'model_sizes.png'))
        plt.close()
<<<<<<< Updated upstream
        
    def _plot_processing_speed(self, speed_results, result_dir):
        """처리 속도 비교 그래프 생성"""
        model_names = []
        speeds = []
        
        for name, speed in speed_results.items():
            model_names.append(name)
            speeds.append(speed)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, speeds)
        plt.title('Processing Speed Comparison')
        plt.xlabel('Model')
        plt.ylabel('Images/sec')
        plt.xticks(rotation=45)
        
        # 막대 위에 속도 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'processing_speed.png'))
        plt.close()
        
    def _generate_summary(self, models_dict, speed_results, result_dir):
        """결과 요약 파일 생성"""
        summary_file = os.path.join(result_dir, 'summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("=== Model Analysis Summary ===\n\n")
            
            # 모델 크기 요약
            f.write("Model Sizes:\n")
            for name, model_info in models_dict.items():
                f.write(f"{name}: {model_info['size']:.2f}MB\n")
            f.write("\n")
            
            # 처리 속도 요약
            f.write("Processing Speed (images/sec):\n")
            for name, speed in speed_results.items():
                f.write(f"{name}: {speed:.2f}\n")
            f.write("\n")
            
            # 압축률 계산
            baseline_size = models_dict['Baseline (ResNet50)']['size']
            for name, model_info in models_dict.items():
                if name != 'Baseline (ResNet50)':
                    compression_ratio = baseline_size / model_info['size']
                    f.write(f"Compression Ratio ({name}/Baseline): {compression_ratio:.2f}x\n")
            
            # 속도 향상률 계산
            baseline_speed = speed_results['Baseline (ResNet50)']
            for name, speed in speed_results.items():
                if name != 'Baseline (ResNet50)':
                    speedup_ratio = speed / baseline_speed
                    f.write(f"Speedup Ratio ({name}/Baseline): {speedup_ratio:.2f}x\n") 
=======

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
            # 모델을 CPU로 이동
            model = model.cpu()
            model.eval()
            
            # 정확도 측정
            correct_top1 = 0
            correct_top5 = 0
            total = 0
            total_time = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    # 데이터를 CPU로 이동
                    images, labels = images.cpu(), labels.cpu()
                    
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
>>>>>>> Stashed changes
