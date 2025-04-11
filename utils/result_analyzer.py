import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from models.custom_quantization_model import CustomQuantization
import torchvision

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