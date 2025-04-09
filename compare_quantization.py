import torch
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from models.baseline_model import SimpleConvNet
from utils.dataset_manager import DatasetManager
from utils.model_evaluator import ModelEvaluator
from utils.inference_benchmark import InferenceBenchmark
from dynamic_quantization import DynamicQuantization
from custom_dynamic_quantization import CustomDynamicQuantization

def apply_quantization_and_compare(model_path):
    """
    학습된 모델에 양자화를 적용하고 성능을 비교하는 함수
    """
    # 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset_manager = DatasetManager()
    _, test_loader, _ = dataset_manager.get_cifar10_dataset(batch_size=64)
    
    # 원본 FP32 모델 로드
    print(f"학습된 FP32 모델 로드 중: {model_path}")
    fp32_model = SimpleConvNet()
    fp32_model.load_state_dict(torch.load(model_path))
    fp32_model.eval()
    
    # 동적 양자화 모델 생성
    print("동적 양자화 모델 생성 중...")
    dynamic_quant = DynamicQuantization(model_path)
    dynamic_model = dynamic_quant.quantize()
    
    # 커스텀 동적 양자화 모델 생성
    print("커스텀 동적 양자화 모델 생성 중...")
    custom_quant = CustomDynamicQuantization(model_path)
    custom_model = custom_quant.quantize()
    
    # 모델 크기 비교
    print("\n모델 크기 비교:")
    fp32_size = dynamic_quant.get_model_size(fp32_model)
    dynamic_size = dynamic_quant.get_model_size(dynamic_model)
    custom_size = custom_quant.get_model_size(custom_model)
    
    print(f"FP32 모델 크기: {fp32_size:.2f} MB")
    print(f"동적 양자화 모델 크기: {dynamic_size:.2f} MB (압축률: {fp32_size/dynamic_size:.2f}x)")
    print(f"커스텀 동적 양자화 모델 크기: {custom_size:.2f} MB (압축률: {fp32_size/custom_size:.2f}x)")
    
    # 모델 정확도 평가
    print("\n모델 정확도 평가:")
    evaluator = ModelEvaluator(test_loader)
    
    print("FP32 모델 정확도 평가 중...")
    fp32_accuracy = evaluator.evaluate_accuracy(fp32_model)
    
    print("동적 양자화 모델 정확도 평가 중...")
    dynamic_accuracy = evaluator.evaluate_accuracy(dynamic_model)
    
    print("커스텀 동적 양자화 모델 정확도 평가 중...")
    custom_accuracy = evaluator.evaluate_accuracy(custom_model)
    
    print("\n정확도 비교:")
    print(f"FP32 모델: {fp32_accuracy:.2f}%")
    print(f"동적 양자화 모델: {dynamic_accuracy:.2f}% (차이: {fp32_accuracy-dynamic_accuracy:.2f}%)")
    print(f"커스텀 동적 양자화 모델: {custom_accuracy:.2f}% (차이: {fp32_accuracy-custom_accuracy:.2f}%)")
    
    # 추론 속도 벤치마크 (1000회 반복)
    print("\n추론 속도 벤치마크 (1000회 반복):")
    benchmark = InferenceBenchmark(test_loader)
    
    # 모델 사전 생성
    models = {
        'FP32 모델': fp32_model,
        '동적 양자화 모델': dynamic_model,
        '커스텀 동적 양자화 모델': custom_model
    }
    
    # 벤치마크 설정 수정 (1000회 반복)
    benchmark.num_iterations = 1000
    benchmark_results = benchmark.compare_models(models)
    
    # 결과 시각화
    visualize_results(fp32_accuracy, dynamic_accuracy, custom_accuracy, 
                     benchmark_results, fp32_size, dynamic_size, custom_size)
    
    # 결과 요약 테이블
    print("\n결과 요약:")
    table_data = [
        ["모델", "정확도 (%)", "모델 크기 (MB)", "압축률", "단일 이미지 추론 (ms)", "배치 처리 추론 (ms/img)", "처리량 (img/sec)"],
        ["FP32 모델", f"{fp32_accuracy:.2f}", f"{fp32_size:.2f}", "1.00x", 
         f"{benchmark_results['FP32 모델']['inference_times']['single_mean']:.2f}", 
         f"{benchmark_results['FP32 모델']['inference_times']['per_image_in_batch']:.2f}", 
         f"{benchmark_results['FP32 모델']['throughput'][32]:.2f}"],
        ["동적 양자화 모델", f"{dynamic_accuracy:.2f}", f"{dynamic_size:.2f}", f"{fp32_size/dynamic_size:.2f}x", 
         f"{benchmark_results['동적 양자화 모델']['inference_times']['single_mean']:.2f}", 
         f"{benchmark_results['동적 양자화 모델']['inference_times']['per_image_in_batch']:.2f}", 
         f"{benchmark_results['동적 양자화 모델']['throughput'][32]:.2f}"],
        ["커스텀 동적 양자화 모델", f"{custom_accuracy:.2f}", f"{custom_size:.2f}", f"{fp32_size/custom_size:.2f}x", 
         f"{benchmark_results['커스텀 동적 양자화 모델']['inference_times']['single_mean']:.2f}", 
         f"{benchmark_results['커스텀 동적 양자화 모델']['inference_times']['per_image_in_batch']:.2f}", 
         f"{benchmark_results['커스텀 동적 양자화 모델']['throughput'][32]:.2f}"]
    ]
    
    print(tabulate(table_data[1:], headers=table_data[0], tablefmt="grid"))
    
    return models, benchmark_results

def visualize_results(fp32_accuracy, dynamic_accuracy, custom_accuracy, 
                     benchmark_results, fp32_size, dynamic_size, custom_size):
    """
    결과를 시각화하는 함수
    """
    # 결과 디렉토리 생성
    os.makedirs("./results", exist_ok=True)
    
    # 1. 정확도 비교 그래프
    model_names = ['FP32 모델', '동적 양자화 모델', '커스텀 동적 양자화 모델']
    accuracies = [fp32_accuracy, dynamic_accuracy, custom_accuracy]
    
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
    
    plt.tight_layout()
    plt.savefig('./results/accuracy_comparison.png')
    plt.close()
    
    # 2. 추론 시간 비교 그래프
    single_times = [benchmark_results[name]['inference_times']['single_mean'] for name in model_names]
    batch_times = [benchmark_results[name]['inference_times']['per_image_in_batch'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, single_times, width, label='단일 이미지')
    rects2 = ax.bar(x + width/2, batch_times, width, label='배치 처리 (이미지당)')
    
    # 막대 위에 값 표시
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    ax.set_title('모델 추론 시간 비교 (1000회 반복)')
    ax.set_ylabel('추론 시간 (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('./results/inference_time_comparison.png')
    plt.close()
    
    # 3. 모델 크기 비교 그래프
    sizes = [fp32_size, dynamic_size, custom_size]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, sizes, color=['blue', 'orange', 'green'])
    
    # 막대 위에 크기 값 표시
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{size:.2f} MB', ha='center', va='bottom')
    
    plt.title('모델 크기 비교')
    plt.ylabel('크기 (MB)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('./results/model_size_comparison.png')
    plt.close()
    
    # 4. 처리량 비교 그래프
    throughputs = [benchmark_results[name]['throughput'][32] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, throughputs, color=['blue', 'orange', 'green'])
    
    # 막대 위에 처리량 값 표시
    for bar, tp in zip(bars, throughputs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{tp:.2f}', ha='center', va='bottom')
    
    plt.title('모델 처리량 비교 (images/sec)')
    plt.ylabel('처리량 (images/sec)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('./results/throughput_comparison.png')
    plt.close()

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='양자화 모델 비교')
    parser.add_argument('--model_path', type=str, default='./saved_models/trained_fp32_model.pth',
                        help='학습된 FP32 모델 경로')
    args = parser.parse_args()
    
    # 모델 경로 확인
    if not os.path.exists(args.model_path):
        print(f"오류: 모델 파일이 존재하지 않습니다: {args.model_path}")
        print("먼저 train_model.py를 실행하여 모델을 학습시켜주세요.")
        return
    
    # 양자화 적용 및 비교
    apply_quantization_and_compare(args.model_path)

if __name__ == "__main__":
    main()

