import torch
import time
import numpy as np
from tqdm import tqdm
import torchvision

class ModelEvaluator:
    """
    모델 정확도를 평가하는 클래스
    """
    def __init__(self, test_loader):
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate_accuracy(self, model, verbose=True):
        """
        모델의 Top-1, Top-5 정확도를 평가하는 함수
        """
        model.eval()
        correct1 = 0
        correct5 = 0
        total = 0
        
        # 양자화된 모델 확인 및 디바이스 설정
        is_quantized = hasattr(model, 'is_quantized') or any(
            isinstance(m, (torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                         torch.ao.nn.quantized.modules.conv.Conv2d))
            for m in model.modules()
        )
        
        if is_quantized:
            model = model.cpu()
            device = torch.device('cpu')
        else:
            device = self.device
            model = model.to(device)
        
        if verbose:
            print(f"\n{'Quantized' if is_quantized else 'FP32'} 모델 평가 중...")
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating", disable=not verbose):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                # Top-1 정확도
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct1 += (predicted == labels).sum().item()
                
                # Top-5 정확도
                _, pred5 = outputs.topk(5, 1, True, True)
                pred5 = pred5.t()
                correct = pred5.eq(labels.view(1, -1).expand_as(pred5))
                correct5 += correct.float().sum().item()
        
        top1_accuracy = 100 * correct1 / total
        top5_accuracy = 100 * correct5 / total
        
        if verbose:
            print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
            print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
        
        return top1_accuracy, top5_accuracy
    
    def evaluate_class_accuracy(self, model, classes, verbose=True):
        """
        클래스별 정확도를 평가하는 함수
        """
        if hasattr(model, 'is_custom_quantized') and model.is_custom_quantized:
            print("Evaluating Custom Quantization Model")
        else:
            print("Evaluating Regular Quantization Model")
        
        model.eval()
        
        # 모델 타입 확인
        if hasattr(model, 'quantized'):
            if hasattr(model, 'is_custom_quantized') and model.is_custom_quantized:
                model_type = "커스텀 동적 양자화 모델"
            else:
                model_type = "일반 동적 양자화 모델"
        else:
            model_type = "FP32 모델"
        
        # 양자화된 모델은 CPU에서 실행
        if hasattr(model, 'quantized'):
            model = model.cpu()
            device = torch.device('cpu')
        else:
            model = model.to(self.device)
            device = self.device
        
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc=f"{model_type} 클래스별 평가 중", disable=not verbose):
                data, target = data.to(device), target.to(device)
                
                # 모델 추론
                output = model(data)
                
                # 예측 결과 계산
                _, predicted = torch.max(output.data, 1)
                
                # 각 클래스별로 정확도 계산
                for i in range(len(target)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        # 클래스별 정확도 계산 및 정렬
        class_accuracies = {}
        for i in range(len(classes)):
            if class_total[i] > 0:  # 해당 클래스의 샘플이 있는 경우에만 계산
                accuracy = 100 * class_correct[i] / class_total[i]
                class_accuracies[classes[i]] = accuracy
        
        # 정확도 기준으로 정렬
        sorted_accuracies = dict(sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True))
        
        if verbose:
            print(f'\n[{model_type}] 클래스별 정확도 (상위 20개 클래스):')
            for i, (class_name, accuracy) in enumerate(sorted_accuracies.items()):
                if i >= 20:
                    break
                print(f'{class_name}: {accuracy:.2f}%')
        
        return sorted_accuracies
    
    def compare_models(self, models_dict, classes):
        """
        여러 모델의 정확도를 비교하는 함수
        """
        results = {}
        
        print("\n=== 모델 정확도 비교 시작 ===")
        for name, model in models_dict.items():
            model_type = "일반 동적 양자화 모델" if hasattr(model, 'quantized') and not hasattr(model, 'is_custom_quantized') else \
                        "커스텀 동적 양자화 모델" if hasattr(model, 'quantized') and model.is_custom_quantized else \
                        "FP32 모델"
            print(f"\n[{model_type}] {name} 모델 평가 중...")
            
            # 모델 평가
            model.eval()
            
            # 양자화된 모델은 CPU에서 실행
            if hasattr(model, 'quantized'):
                model = model.cpu()
                device = torch.device('cpu')
            else:
                model = model.to(self.device)
                device = self.device
            
            correct = 0
            total = 0
            class_correct = list(0. for i in range(len(classes)))
            class_total = list(0. for i in range(len(classes)))
            
            with torch.no_grad():
                for data, target in tqdm(self.test_loader, desc=f"{model_type} 평가 중"):
                    data, target = data.to(device), target.to(device)
                    
                    # 모델 추론
                    output = model(data)
                    
                    # 예측 결과 계산
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    
                    # 각 클래스별로 정확도 계산
                    for i in range(len(target)):
                        label = target[i]
                        class_correct[label] += (predicted[i] == label).item()
                        class_total[label] += 1
            
            # 전체 정확도 계산
            accuracy = 100 * correct / total
            
            # 클래스별 정확도 계산 및 정렬
            class_accuracies = {}
            for i in range(len(classes)):
                if class_total[i] > 0:  # 해당 클래스의 샘플이 있는 경우에만 계산
                    class_accuracy = 100 * class_correct[i] / class_total[i]
                    class_accuracies[classes[i]] = class_accuracy
            
            # 정확도 기준으로 정렬
            sorted_accuracies = dict(sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True))
            
            results[name] = {
                'accuracy': accuracy,
                'class_accuracies': sorted_accuracies
            }
            
            # 결과 출력
            print(f"[{model_type}] {name} 정확도: {accuracy:.2f}%")
            print(f"\n[{model_type}] 클래스별 정확도 (상위 20개 클래스):")
            for i, (class_name, class_accuracy) in enumerate(sorted_accuracies.items()):
                if i >= 20:
                    break
                print(f'{class_name}: {class_accuracy:.2f}%')
        
        # 결과 요약
        print("\n=== 모델 정확도 비교 결과 ===")
        for name, result in results.items():
            model_type = "일반 동적 양자화 모델" if hasattr(models_dict[name], 'quantized') and not hasattr(models_dict[name], 'is_custom_quantized') else \
                        "커스텀 동적 양자화 모델" if hasattr(models_dict[name], 'quantized') and models_dict[name].is_custom_quantized else \
                        "FP32 모델"
            print(f"[{model_type}] {name}: {result['accuracy']:.2f}%")
        
        return results

# 테스트 함수
def test_evaluator():
    from utils.dataset_manager import DatasetManager
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # 평가기 생성
    evaluator = ModelEvaluator(test_loader)
    
    # 테스트용 모델 생성 (FP32)
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 정확도 평가
    print("\nFP32 모델 정확도 평가:")
    top1, top5 = evaluator.evaluate_accuracy(model)
    
    return evaluator

if __name__ == "__main__":
    test_evaluator()

