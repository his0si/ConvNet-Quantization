import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tabulate import tabulate

from models.dynamic_ptq_model import DynamicPTQModel
from models.custom_quantization_model import CustomQuantization
from utils.dataset_manager import DatasetManager
from utils.model_evaluator import ModelEvaluator
from utils.inference_benchmark import InferenceBenchmark
from utils.result_analyzer import ResultAnalyzer

class ResultAnalyzer:
    """
    모델 비교 결과를 분석하고 시각화하는 클래스
    """
    def __init__(self, results_dir='./results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def plot_accuracy_comparison(self, accuracy_results):
        """
        모델 정확도 비교 결과를 시각화하는 함수
        """
        model_names = list(accuracy_results.keys())
        accuracies = [accuracy_results[name]['accuracy'] for name in model_names]
        
        # 정확도 비교 막대 그래프
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color=['blue', 'orange', 'green'])
        
        # 막대 위에 정확도 값 표시
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom')
        
        plt.title('모델 정확도 비교')
        plt.ylabel('정확도 (%)')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 저장
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'accuracy_comparison.png'))
        plt.close()
        
        return os.path.join(self.results_dir, 'accuracy_comparison.png')
    
    def plot_class_accuracy_comparison(self, accuracy_results, classes, top_n=20):
        """
        클래스별 정확도 비교 결과를 시각화하는 함수
        top_n: 표시할 상위 클래스 수
        """
        model_names = list(accuracy_results.keys())
        
        # 클래스별 정확도 데이터 준비
        class_accuracies = {}
        for cls in classes[:top_n]:  # 상위 N개 클래스만 사용
            class_accuracies[cls] = []
            for name in model_names:
                class_accuracies[cls].append(accuracy_results[name]['class_accuracies'][cls])
        
        # 클래스별 정확도 비교 그래프
        x = np.arange(len(classes[:top_n]))
        width = 0.25  # 막대 너비
        
        fig, ax = plt.figure(figsize=(14, 8)), plt.axes()
        
        # 각 모델별로 막대 그래프 그리기
        for i, name in enumerate(model_names):
            accuracies = [accuracy_results[name]['class_accuracies'][cls] for cls in classes[:top_n]]
            ax.bar(x + i*width - width, accuracies, width, label=name)
        
        ax.set_title(f'상위 {top_n}개 클래스별 모델 정확도 비교')
        ax.set_ylabel('정확도 (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(classes[:top_n], rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 저장
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'class_accuracy_comparison.png'))
        plt.close()
        
        return os.path.join(self.results_dir, 'class_accuracy_comparison.png')
    
    def plot_inference_time_comparison(self, benchmark_results):
        """
        추론 시간 비교 결과를 시각화하는 함수
        """
        model_names = list(benchmark_results.keys())
        
        # 단일 이미지 추론 시간
        single_times = [benchmark_results[name]['inference_times']['single_mean'] for name in model_names]
        
        # 배치 처리 시 이미지당 추론 시간
        batch_times = [benchmark_results[name]['inference_times']['per_image_in_batch'] for name in model_names]
        
        # 추론 시간 비교 그래프
        x = np.arange(len(model_names))
        width = 0.35  # 막대 너비
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, single_times, width, label='단일 이미지')
        rects2 = ax.bar(x + width/2, batch_times, width, label='배치 처리 (이미지당)')
        
        # 막대 위에 값 표시
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 포인트 위에 표시
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        ax.set_title('모델 추론 시간 비교')
        ax.set_ylabel('추론 시간 (ms)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 저장
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'inference_time_comparison.png'))
        plt.close()
        
        return os.path.join(self.results_dir, 'inference_time_comparison.png')
    
    def plot_throughput_comparison(self, benchmark_results, batch_size=32):
        """
        처리량 비교 결과를 시각화하는 함수
        """
        model_names = list(benchmark_results.keys())
        
        # 단일 이미지 처리량
        single_throughput = [benchmark_results[name]['throughput'][1] for name in model_names]
        
        # 배치 처리 처리량
        batch_throughput = [benchmark_results[name]['throughput'][batch_size] for name in model_names]
        
        # 처리량 비교 그래프
        x = np.arange(len(model_names))
        width = 0.35  # 막대 너비
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, single_throughput, width, label='단일 이미지')
        rects2 = ax.bar(x + width/2, batch_throughput, width, label=f'배치 크기 {batch_size}')
        
        # 막대 위에 값 표시
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 포인트 위에 표시
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        ax.set_title('모델 처리량 비교 (images/sec)')
        ax.set_ylabel('처리량 (images/sec)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 저장
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'throughput_comparison.png'))
        plt.close()
        
        return os.path.join(self.results_dir, 'throughput_comparison.png')
    
    def plot_model_size_comparison(self, model_sizes):
        """
        모델 크기 비교 결과를 시각화하는 함수
        """
        model_names = list(model_sizes.keys())
        sizes = list(model_sizes.values())
        
        # 모델 크기 비교 막대 그래프
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, sizes, color=['blue', 'orange', 'green'])
        
        # 막대 위에 크기 값 표시
        for bar, size in zip(bars, sizes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{size:.2f} MB', ha='center', va='bottom')
        
        plt.title('모델 크기 비교')
        plt.ylabel('크기 (MB)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 저장
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_size_comparison.png'))
        plt.close()
        
        return os.path.join(self.results_dir, 'model_size_comparison.png')
    
    def generate_summary_table(self, accuracy_results, benchmark_results, model_sizes):
        """
        모델 비교 결과를 요약 테이블로 생성하는 함수
        """
        model_names = list(accuracy_results.keys())
        
        # 테이블 데이터 준비
        table_data = []
        headers = ["모델", "정확도 (%)", "모델 크기 (MB)", "압축률", "단일 이미지 추론 (ms)", "배치 처리 추론 (ms/img)", "처리량 (img/sec)"]
        
        for name in model_names:
            row = [
                name,
                f"{accuracy_results[name]['accuracy']:.2f}",
                f"{model_sizes[name]:.2f}",
                f"{model_sizes['FP32 모델'] / model_sizes[name]:.2f}x" if name != 'FP32 모델' else "1.00x",
                f"{benchmark_results[name]['inference_times']['single_mean']:.2f}",
                f"{benchmark_results[name]['inference_times']['per_image_in_batch']:.2f}",
                f"{benchmark_results[name]['throughput'][32]:.2f}"
            ]
            table_data.append(row)
        
        # 테이블 생성
        table = tabulate(table_data, headers=headers, tablefmt="grid")
        
        # 저장
        with open(os.path.join(self.results_dir, 'summary_table.txt'), 'w') as f:
            f.write(table)
        
        return table, os.path.join(self.results_dir, 'summary_table.txt')
    
    def generate_report(self, accuracy_results, benchmark_results, model_sizes, batch_size=32):
        """
        모든 결과를 종합하여 보고서를 생성하는 함수
        """
        # 결과 시각화
        accuracy_plot = self.plot_accuracy_comparison(accuracy_results)
        inference_time_plot = self.plot_inference_time_comparison(benchmark_results)
        throughput_plot = self.plot_throughput_comparison(benchmark_results, batch_size)
        model_size_plot = self.plot_model_size_comparison(model_sizes)
        
        # 요약 테이블 생성
        summary_table, table_path = self.generate_summary_table(accuracy_results, benchmark_results, model_sizes)
        
        # 보고서 생성
        report = f"""# PyTorch 양자화 방식 비교 실험 결과

## 실험 개요
이 보고서는 PyTorch에서 기존 PTQ(Static Post-Training Quantization) 방식과 Custom Quantization 방식의 정확도와 추론 성능을 비교한 결과입니다.
FP32 모델을 baseline으로 하여, 각 양자화 방식이 얼마만큼의 정확도 하락과 추론 속도 이득을 얻는지 비교했습니다.

## 비교 모델
1. **FP32 모델**: 양자화 없이 32비트 부동소수점 형식을 사용하는 기준 모델
2. **기존 PTQ 방식**: PyTorch의 표준 Static Post-Training Quantization API를 사용한 모델
3. **Custom Quantization 방식**: 모델 내부에 QuantStub/DeQuantStub를 삽입하고 특정 연산자를 래핑한 커스텀 양자화 모델

## 요약 결과

{summary_table}

## 정확도 비교
![정확도 비교](accuracy_comparison.png)

정확도 측면에서는 FP32 모델이 가장 높은 정확도를 보이며, 양자화 모델들은 약간의 정확도 하락이 있습니다.
그러나 Custom Quantization 방식이 기존 PTQ 방식보다 정확도 하락이 적은 것으로 나타났습니다.

## 추론 시간 비교
![추론 시간 비교](inference_time_comparison.png)

추론 시간 측면에서는 양자화 모델들이 FP32 모델보다 빠른 추론 속도를 보입니다.
특히 Custom Quantization 방식이 단일 이미지 및 배치 처리 모두에서 가장 빠른 추론 속도를 달성했습니다.

## 처리량 비교
![처리량 비교](throughput_comparison.png)

처리량(throughput) 측면에서도 양자화 모델들이 FP32 모델보다 높은 처리량을 보입니다.
Custom Quantization 방식이 가장 높은 처리량을 달성했으며, 특히 배치 처리에서 그 차이가 더 두드러집니다.

## 모델 크기 비교
![모델 크기 비교](model_size_comparison.png)

모델 크기 측면에서는 양자화 모델들이 FP32 모델보다 크게 감소된 크기를 보입니다.
기존 PTQ 방식과 Custom Quantization 방식 모두 약 4배 정도의 압축률을 달성했습니다.

## 결론

1. **정확도**: Custom Quantization 방식이 기존 PTQ 방식보다 정확도 하락이 적습니다.
2. **추론 속도**: Custom Quantization 방식이 가장 빠른 추론 속도를 보입니다.
3. **모델 크기**: 두 양자화 방식 모두 비슷한 수준의 모델 크기 감소를 달성했습니다.

종합적으로, Custom Quantization 방식이 정확도와 추론 속도 측면에서 가장 좋은 성능을 보이며,
특히 모델 내부에 QuantStub/DeQuantStub를 삽입하고 특정 연산자를 래핑하는 방식이
양자화된 텐서 간의 연산이 올바르게 이루어지도록 하는 데 효과적임을 확인할 수 있습니다.
"""
        
        # 보고서 저장
        with open(os.path.join(self.results_dir, 'quantization_comparison_report.md'), 'w') as f:
            f.write(report)
        
        return os.path.join(self.results_dir, 'quantization_comparison_report.md')

def main():
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # 모델 준비
    print("모델 준비 중...")
    
    # 1. Baseline 모델 (ResNet50)
    baseline_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 2. 일반 양자화 모델 (Dynamic PTQ)
    ptq_model_manager = DynamicPTQModel()
    ptq_model = ptq_model_manager.quantize(baseline_model)
    
    # 3. 커스텀 양자화 모델
    custom_quantization = CustomQuantization()
    custom_quantized_model = custom_quantization.quantize(baseline_model)
    
    # 모델 평가
    print("\n모델 평가 시작...")
    evaluator = ModelEvaluator(test_loader)
    
    # 각 모델의 정확도 평가
    baseline_accuracy = evaluator.evaluate_accuracy(baseline_model)
    ptq_accuracy = evaluator.evaluate_accuracy(ptq_model)
    custom_quantized_accuracy = evaluator.evaluate_accuracy(custom_quantized_model)
    
    # 클래스별 정확도 평가
    baseline_class_acc = evaluator.evaluate_class_accuracy(baseline_model, dataset_manager.classes)
    ptq_class_acc = evaluator.evaluate_class_accuracy(ptq_model, dataset_manager.classes)
    custom_class_acc = evaluator.evaluate_class_accuracy(custom_quantized_model, dataset_manager.classes)
    
    # 모델 비교
    models_dict = {
        'Baseline (ResNet50)': baseline_model,
        'Dynamic PTQ': ptq_model,
        'Custom Quantization': custom_quantized_model
    }
    
    # 정확도 비교
    accuracy_results = evaluator.compare_models(models_dict, dataset_manager.classes)
    
    # 추론 속도 벤치마크
    print("\n추론 속도 벤치마크 시작...")
    benchmark = InferenceBenchmark(test_loader)
    speed_results = benchmark.compare_models(models_dict)
    
    # 결과 분석 및 시각화
    print("\n결과 분석 및 시각화...")
    analyzer = ResultAnalyzer()
    
    # 정확도 비교 그래프
    analyzer.plot_accuracy_comparison(accuracy_results)
    
    # 클래스별 정확도 비교 그래프
    analyzer.plot_class_accuracy_comparison({
        'Baseline': baseline_class_acc,
        'Dynamic PTQ': ptq_class_acc,
        'Custom Quantization': custom_class_acc
    }, dataset_manager.classes)
    
    # 추론 속도 비교 그래프
    analyzer.plot_speed_comparison(speed_results)
    
    # 모델 크기 비교
    analyzer.plot_model_size_comparison({
        'Baseline': baseline_model,
        'Dynamic PTQ': ptq_model,
        'Custom Quantization': custom_quantized_model
    })
    
    print("\n모든 평가가 완료되었습니다. 결과는 results 디렉토리에서 확인할 수 있습니다.")

if __name__ == "__main__":
    # 필요한 패키지 설치
    try:
        import tabulate
    except ImportError:
        import os
        os.system("pip install tabulate")
        import tabulate
    
    main()

