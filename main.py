import torch
<<<<<<< Updated upstream
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
=======
import torch.nn as nn
from models.baseline_model import SimpleConvNet
from models.dynamic_ptq_model import DynamicPTQModel
from models.custom_quantization_model import CustomQuantizationModel
from utils.dataset_manager import DatasetManager
from utils.model_evaluator import ModelEvaluator
from utils.result_analyzer import ResultAnalyzer
>>>>>>> Stashed changes
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
    print("모델 생성 및 양자화 중...")
    
    # 1. Baseline 모델 (ResNet50)
    print("- Baseline 모델 (ResNet50) 생성 중...")
    baseline = BaselineModel()
    baseline_model = baseline.get_model()
    baseline_model = baseline_model.cpu()  # CPU로 이동
    
    # 2. 일반 양자화 모델 (Dynamic PTQ)
    print("- Dynamic PTQ 모델 생성 중...")
    ptq_model_manager = DynamicPTQModel()
    ptq_model = ptq_model_manager.quantize(baseline_model)
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
    
    # 추론 속도 벤치마크
    print("\n추론 속도 벤치마크 시작...")
    benchmark = InferenceBenchmark(test_loader)
    speed_results = benchmark.compare_models(models_dict)
    
<<<<<<< Updated upstream
    # 결과 분석 및 시각화
    print("\n결과 분석 및 시각화...")
    analyzer = ResultAnalyzer()
    analyzer.analyze_results(models_dict, speed_results)
    
    # 모델 크기 비교
    print("\n모델 크기 비교:")
    for name, model in models_dict.items():
        size_mb = custom_quantization.get_model_size(model)
        print(f"{name}: {size_mb:.2f} MB")
    
    # 정확도 평가
    print("\n정확도 평가:")
    for name, model in models_dict.items():
        top1, top5 = dataset_manager.evaluate_model(model)
        print(f"{name}:")
        print(f"  Top-1 Accuracy: {top1:.2f}%")
        print(f"  Top-5 Accuracy: {top5:.2f}%")
=======
    # 결과 출력
    print("\n=== 최종 결과 ===")
    for name, (top1, top5) in results.items():
        print(f"\n{name}:")
        print(f"Top-1 Accuracy: {top1:.2f}%")
        print(f"Top-5 Accuracy: {top5:.2f}%")
    
    # 양자화 방법 비교 분석
    print("\n=== 양자화 방법 비교 분석 ===")
    analyzer = ResultAnalyzer()
    comparison_results = analyzer.compare_quantization_methods(
        fp32_model=baseline_model,
        ptq_model=dynamic_ptq_model,
        proposed_model=custom_quant_model,
        test_loader=test_loader
    )
    
    print("\n분석 결과가 'quantization_comparison.png'와 'quantization_comparison.csv'에 저장되었습니다.")
>>>>>>> Stashed changes

if __name__ == "__main__":
    # 필요한 패키지 설치
    try:
        import tabulate
    except ImportError:
        import os
        os.system("pip install tabulate")
        import tabulate
    
    main()

