import torch
import time
import numpy as np
from tqdm import tqdm

class ModelEvaluator:
    """
    모델 정확도를 평가하는 클래스
    """
    def __init__(self, test_loader):
        self.test_loader = test_loader
        # GPU 환경에서 실행하려면 아래 줄을 수정하세요
        self.device = torch.device("cpu")  # CPU 기반 평가
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용
    
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
                c = (predicted == target).squeeze()
                
                for i in range(len(target)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        # 클래스별 정확도 계산
        class_accuracies = {}
        if verbose:
            print('\n클래스별 정확도:')
        for i in range(len(classes)):
            accuracy = 100 * class_correct[i] / class_total[i]
            class_accuracies[classes[i]] = accuracy
            if verbose:
                print(f'{classes[i]}: {accuracy:.2f}%')
        
        return class_accuracies
    
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
    from models.baseline_model import create_model
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    _, test_loader, _ = dataset_manager.get_cifar10_dataset(batch_size=64)
    
    # 평가기 생성
    evaluator = ModelEvaluator(test_loader)
    
    # 테스트용 모델 생성
    model = create_model()
    
    # 정확도 평가 (학습되지 않은 모델이므로 약 10% 정도의 정확도가 예상됨)
    print("모델 정확도 평가 테스트:")
    accuracy = evaluator.evaluate_accuracy(model)
    
    # 클래스별 정확도 평가
    print("\n클래스별 정확도 평가 테스트:")
    class_accuracies = evaluator.evaluate_class_accuracy(model, dataset_manager.classes)
    
    return evaluator

if __name__ == "__main__":
    test_evaluator()

