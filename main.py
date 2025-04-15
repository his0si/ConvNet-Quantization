import torch
import torch.nn as nn
from models.baseline_model import SimpleConvNet
from models.dynamic_ptq_model import DynamicPTQModel
from models.custom_quantization_model import CustomQuantizationModel
from utils.dataset_manager import DatasetManager
from utils.model_evaluator import ModelEvaluator
from utils.result_analyzer import ResultAnalyzer
import os

def load_trained_model(model_path='./trained_model.pth'):
    """
    학습된 모델을 로드하는 함수
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"학습된 모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 모델 생성
    model = SimpleConvNet()
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"학습된 모델을 {model_path}에서 로드했습니다.")
    print(f"최고 정확도: {checkpoint['best_accuracy']:.2f}%")
    
    return model

def apply_to_models(trained_model):
    """
    학습된 모델을 세 가지 모델에 적용하는 함수
    """
    # 1. 기본 모델
    print("\n=== 기본 모델 ===")
    baseline_model = SimpleConvNet()
    baseline_model.load_state_dict(trained_model.state_dict())
    
    # 2. 동적 양자화 모델
    print("\n=== 동적 양자화 모델 ===")
    dynamic_ptq_model = DynamicPTQModel()
    dynamic_ptq_model.load_state_dict(trained_model.state_dict())
    dynamic_ptq_model.quantize()
    
    # 3. 커스텀 양자화 모델
    print("\n=== 커스텀 양자화 모델 ===")
    custom_quant_model = CustomQuantizationModel()
    custom_quant_model.load_state_dict(trained_model.state_dict())
    custom_quant_model.quantize()
    
    return baseline_model, dynamic_ptq_model, custom_quant_model

def evaluate_models(models, test_loader):
    """
    세 가지 모델을 평가하는 함수
    """
    evaluator = ModelEvaluator(test_loader)
    results = {}
    
    for name, model in models.items():
        print(f"\n=== {name} 평가 ===")
        top1, top5 = evaluator.evaluate_accuracy(model)
        results[name] = (top1, top5)
        print(f"{name} 정확도:")
        print(f"Top-1 Accuracy: {top1:.2f}%")
        print(f"Top-5 Accuracy: {top5:.2f}%")
    
    return results

def main():
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    train_loader, test_loader, classes = dataset_manager.get_cifar10_dataset(batch_size=128)
    
    # 학습된 모델 로드
    trained_model = load_trained_model()
    
    # 세 가지 모델에 적용
    baseline_model, dynamic_ptq_model, custom_quant_model = apply_to_models(trained_model)
    
    # 모델 평가
    models = {
        "기본 모델": baseline_model,
        "동적 양자화 모델": dynamic_ptq_model,
        "커스텀 양자화 모델": custom_quant_model
    }
    
    results = evaluate_models(models, test_loader)
    
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

if __name__ == "__main__":
    main()

