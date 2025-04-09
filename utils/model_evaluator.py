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
        모델의 정확도를 평가하는 함수
        """
        model.eval()
        model = model.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="평가 중", disable=not verbose):
                data, target = data.to(self.device), target.to(self.device)
                
                # 모델 추론
                output = model(data)
                
                # 예측 결과 계산
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        if verbose:
            print(f'정확도: {accuracy:.2f}%')
        
        return accuracy
    
    def evaluate_class_accuracy(self, model, classes, verbose=True):
        """
        클래스별 정확도를 평가하는 함수
        """
        model.eval()
        model = model.to(self.device)
        
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="클래스별 평가 중", disable=not verbose):
                data, target = data.to(self.device), target.to(self.device)
                
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
            print('\n클래스별 정확도 (상위 20개 클래스):')
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
        
        print("모델 정확도 비교 시작...")
        for name, model in models_dict.items():
            print(f"\n{name} 모델 평가 중...")
            accuracy = self.evaluate_accuracy(model)
            class_accuracies = self.evaluate_class_accuracy(model, classes)
            
            results[name] = {
                'accuracy': accuracy,
                'class_accuracies': class_accuracies
            }
        
        # 결과 요약
        print("\n=== 모델 정확도 비교 결과 ===")
        for name, result in results.items():
            print(f"{name}: {result['accuracy']:.2f}%")
        
        return results

# 테스트 함수
def test_evaluator():
    from utils.dataset_manager import DatasetManager
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # 평가기 생성
    evaluator = ModelEvaluator(test_loader)
    
    # 테스트용 모델 생성
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 정확도 평가
    print("모델 정확도 평가 테스트:")
    accuracy = evaluator.evaluate_accuracy(model)
    
    # 클래스별 정확도 평가
    print("\n클래스별 정확도 평가 테스트:")
    class_accuracies = evaluator.evaluate_class_accuracy(model, dataset_manager.classes)
    
    return evaluator

if __name__ == "__main__":
    test_evaluator()

