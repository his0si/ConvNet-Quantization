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
from models.baseline_model import BaselineModel
from utils.dataset_manager import DatasetManager
from utils.inference_benchmark import InferenceBenchmark
from utils.result_analyzer import ResultAnalyzer

def main():
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset(batch_size=64)
    
    # 모델 준비
    print("\n모델 생성 및 양자화 중...")
    
    # 1. Baseline 모델 (ResNet50)
    print("- Baseline 모델 (ResNet50) 생성 중...")
    baseline = BaselineModel()
    baseline_model = baseline.get_model()
    baseline_model = baseline_model.cpu()  # CPU로 이동
    
    # 2. 일반 양자화 모델 (Dynamic PTQ)
    print("- Dynamic PTQ 모델 생성 중...")
    ptq_model_manager = DynamicPTQModel()
    ptq_model = ptq_model_manager.quantize()
    ptq_model = ptq_model.cpu()  # CPU로 이동
    
    # 3. 커스텀 양자화 모델
    print("- Custom 양자화 모델 생성 중...")
    custom_quantization = CustomQuantization(baseline_model)
    custom_quantized_model = custom_quantization.quantize()
    custom_quantized_model = custom_quantized_model.cpu()  # CPU로 이동
    
    print("모델 생성 완료!")
    
    # 모델 비교를 위한 딕셔너리
    models_dict = {
        'Baseline (ResNet50)': baseline_model,
        'Dynamic PTQ': ptq_model,
        'Custom Quantization': custom_quantized_model
    }
    
    # 결과 저장용 딕셔너리
    results = {
        'accuracy': {},
        'model_size': {},
        'inference_speed': {}
    }
    
    # 1. 정확도 평가
    print("\n정확도 평가 중...")
    for name, model in models_dict.items():
        print(f"\n{name} 평가 중...")
        top1, top5 = dataset_manager.evaluate_model(model)
        results['accuracy'][name] = {'top1': top1, 'top5': top5}
        print(f"Top-1 Accuracy: {top1:.2f}%")
        print(f"Top-5 Accuracy: {top5:.2f}%")
    
    # 2. 모델 크기 측정
    print("\n모델 크기 측정 중...")
    for name, model in models_dict.items():
        size_mb = custom_quantization.get_model_size()
        results['model_size'][name] = size_mb
        print(f"{name}: {size_mb:.2f} MB")
    
    # 3. 추론 속도 측정
    print("\n추론 속도 측정 중...")
    benchmark = InferenceBenchmark(test_loader)
    speed_results = benchmark.compare_models(models_dict)
    results['inference_speed'] = speed_results
    
    # 4. 결과 분석 및 시각화
    print("\n결과 분석 및 시각화 중...")
    analyzer = ResultAnalyzer()
    analyzer.analyze_and_plot(results)
    
    # 결과 요약 출력
    print("\n=== 최종 결과 요약 ===")
    print("\n1. 정확도 (Top-1 / Top-5):")
    for name in models_dict.keys():
        acc = results['accuracy'][name]
        print(f"{name}: {acc['top1']:.2f}% / {acc['top5']:.2f}%")
    
    print("\n2. 모델 크기:")
    for name in models_dict.keys():
        print(f"{name}: {results['model_size'][name]:.2f} MB")
    
    print("\n3. 추론 속도 (images/sec):")
    for name in models_dict.keys():
        print(f"{name}: {results['inference_speed'][name]:.2f}")
    
    print("\n분석 완료! 상세 결과는 'model_comparison.png' 파일에서 확인할 수 있습니다.")

if __name__ == "__main__":
    # 필요한 패키지 설치
    try:
        import tabulate
    except ImportError:
        import os
        os.system("pip install tabulate")
        import tabulate
    
    main()

