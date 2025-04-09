import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import tabulate

class ResultAnalyzer:
    """
    실험 결과를 분석하고 시각화하는 클래스
    """
    def __init__(self):
        # 결과 저장 디렉토리 생성
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def plot_accuracy_comparison(self, accuracy_results):
        """
        모델별 정확도를 비교하는 그래프를 생성하는 함수
        """
        models = list(accuracy_results.keys())
        accuracies = [accuracy_results[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies)
        
        # 각 막대 위에 정확도 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')
        
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 그래프 저장
        plt.savefig(os.path.join(self.results_dir, 'accuracy_comparison.png'))
        plt.close()
    
    def plot_class_accuracy_comparison(self, accuracy_results, classes, top_n=20):
        """
        클래스별 정확도를 비교하는 그래프를 생성하는 함수
        """
        plt.figure(figsize=(15, 8))
        
        # 각 모델의 클래스별 정확도를 정렬하여 상위 N개 선택
        for model_name, class_acc in accuracy_results.items():
            sorted_classes = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)[:top_n]
            class_names = [c[0] for c in sorted_classes]
            accuracies = [c[1] for c in sorted_classes]
            
            plt.plot(class_names, accuracies, marker='o', label=model_name)
        
        plt.title(f'Class-wise Accuracy Comparison (Top {top_n} Classes)')
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 그래프 저장
        plt.savefig(os.path.join(self.results_dir, 'class_accuracy_comparison.png'))
        plt.close()
    
    def plot_speed_comparison(self, speed_results):
        """
        모델별 추론 속도를 비교하는 그래프를 생성하는 함수
        """
        models = list(speed_results.keys())
        
        # 단일 이미지 추론 시간
        single_times = [speed_results[model]['inference_times']['single_mean'] for model in models]
        
        # 배치 처리 시 이미지당 추론 시간
        batch_times = [speed_results[model]['inference_times']['per_image_in_batch'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        rects1 = plt.bar(x - width/2, single_times, width, label='Single Image')
        rects2 = plt.bar(x + width/2, batch_times, width, label='Per Image in Batch')
        
        # 각 막대 위에 시간 값 표시
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.2f}ms',
                        ha='center', va='bottom')
        
        plt.title('Model Inference Speed Comparison')
        plt.xlabel('Model')
        plt.ylabel('Inference Time (ms)')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 그래프 저장
        plt.savefig(os.path.join(self.results_dir, 'speed_comparison.png'))
        plt.close()
    
    def plot_model_size_comparison(self, models_dict):
        """
        모델별 크기를 비교하는 그래프를 생성하는 함수
        """
        def get_model_size(model):
            torch.save(model.state_dict(), "temp_model.pth")
            size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
            os.remove("temp_model.pth")
            return size_mb
        
        models = list(models_dict.keys())
        sizes = [get_model_size(model) for model in models_dict.values()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, sizes)
        
        # 각 막대 위에 크기 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}MB',
                    ha='center', va='bottom')
        
        plt.title('Model Size Comparison')
        plt.xlabel('Model')
        plt.ylabel('Size (MB)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 그래프 저장
        plt.savefig(os.path.join(self.results_dir, 'size_comparison.png'))
        plt.close()
    
    def generate_report(self, accuracy_results, speed_results, model_sizes):
        """
        실험 결과를 종합한 보고서를 생성하는 함수
        """
        # 정확도 결과 테이블
        accuracy_table = []
        for model, results in accuracy_results.items():
            accuracy_table.append([
                model,
                f"{results['accuracy']:.2f}%",
                f"{results.get('top5_accuracy', 'N/A')}%"
            ])
        
        # 속도 결과 테이블
        speed_table = []
        for model, results in speed_results.items():
            speed_table.append([
                model,
                f"{results['inference_times']['single_mean']:.2f}ms",
                f"{results['inference_times']['per_image_in_batch']:.2f}ms",
                f"{results['throughput'][32]:.2f} images/sec"
            ])
        
        # 모델 크기 테이블
        size_table = []
        for model, size in model_sizes.items():
            size_table.append([model, f"{size:.2f}MB"])
        
        # 보고서 생성
        report = f"""# PyTorch Quantization Comparison Report

## 1. Model Accuracy

{tabulate.tabulate(accuracy_table, headers=['Model', 'Top-1 Accuracy', 'Top-5 Accuracy'], tablefmt='pipe')}

## 2. Inference Speed

{tabulate.tabulate(speed_table, headers=['Model', 'Single Image', 'Per Image in Batch', 'Throughput (32 batch)'], tablefmt='pipe')}

## 3. Model Size

{tabulate.tabulate(size_table, headers=['Model', 'Size'], tablefmt='pipe')}

## 4. Visualization

- Accuracy Comparison: accuracy_comparison.png
- Class-wise Accuracy: class_accuracy_comparison.png
- Speed Comparison: speed_comparison.png
- Size Comparison: size_comparison.png
"""
        
        # 보고서 저장
        report_path = os.path.join(self.results_dir, 'quantization_comparison_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path 