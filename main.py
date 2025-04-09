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
from utils.inference_benchmark import InferenceBenchmark
from utils.result_analyzer import ResultAnalyzer

def main():
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # 모델 준비
    print("모델 생성 및 양자화 중...")
    
    # 1. Baseline 모델 (ResNet50)
    print("- Baseline 모델 (ResNet50) 생성 중...")
    baseline_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 2. 일반 양자화 모델 (Dynamic PTQ)
    print("- Dynamic PTQ 모델 생성 중...")
    ptq_model_manager = DynamicPTQModel()
    ptq_model = ptq_model_manager.quantize(baseline_model)
    
    # 3. 커스텀 양자화 모델
    print("- Custom 양자화 모델 생성 중...")
    custom_quantization = CustomQuantization()
    custom_quantized_model = custom_quantization.quantize(baseline_model)
    
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
    
    # 결과 분석 및 시각화
    print("\n결과 분석 및 시각화...")
    analyzer = ResultAnalyzer()
    
    # 정확도 비교 그래프
    analyzer.plot_accuracy_comparison({
        'Baseline (ResNet50)': baseline_model.accuracy,
        'Dynamic PTQ': ptq_model.accuracy,
        'Custom Quantization': custom_quantized_model.accuracy
    })
    
    # 클래스별 정확도 비교 그래프
    analyzer.plot_class_accuracy_comparison({
        'Baseline (ResNet50)': baseline_model.class_accuracy,
        'Dynamic PTQ': ptq_model.class_accuracy,
        'Custom Quantization': custom_quantized_model.class_accuracy
    }, dataset_manager.classes)
    
    # 추론 속도 비교 그래프
    analyzer.plot_speed_comparison(speed_results)
    
    # 모델 크기 비교
    analyzer.plot_model_size_comparison(models_dict)
    
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

