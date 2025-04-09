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

