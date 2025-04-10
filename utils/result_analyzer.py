import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class ResultAnalyzer:
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
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
        
        # 모델 크기 비교
        self._plot_model_sizes(models_dict, result_dir)
        
        # 처리 속도 비교
        self._plot_processing_speed(speed_results, result_dir)
        
        # 결과 요약 파일 생성
        self._generate_summary(models_dict, speed_results, result_dir)
        
    def _plot_model_sizes(self, models_dict, result_dir):
        """모델 크기 비교 그래프 생성"""
        model_names = []
        sizes = []
        
        for name, model in models_dict.items():
            model_names.append(name)
            sizes.append(model.get('size', 0))
        
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
            for name, model in models_dict.items():
                f.write(f"{name}: {model.get('size', 0):.2f}MB\n")
            f.write("\n")
            
            # 처리 속도 요약
            f.write("Processing Speed (images/sec):\n")
            for name, speed in speed_results.items():
                f.write(f"{name}: {speed:.2f}\n")
            f.write("\n")
            
            # 압축률 계산
            if 'FP32' in models_dict and 'INT8' in models_dict:
                fp32_size = models_dict['FP32'].get('size', 0)
                int8_size = models_dict['INT8'].get('size', 0)
                compression_ratio = fp32_size / int8_size if int8_size > 0 else 0
                f.write(f"Compression Ratio (FP32/INT8): {compression_ratio:.2f}x\n")
            
            # 속도 향상률 계산
            if 'FP32' in speed_results and 'INT8' in speed_results:
                fp32_speed = speed_results['FP32']
                int8_speed = speed_results['INT8']
                speedup_ratio = int8_speed / fp32_speed if fp32_speed > 0 else 0
                f.write(f"Speedup Ratio (INT8/FP32): {speedup_ratio:.2f}x\n") 