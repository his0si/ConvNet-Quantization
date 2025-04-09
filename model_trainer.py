import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm

class ModelTrainer:
    """
    모델 학습을 위한 클래스
    """
    def __init__(self, model, train_loader, test_loader, device=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 학습 결과 저장
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
    
    def train(self, epochs=10, lr=0.001, save_path='./trained_model.pth'):
        """
        모델 학습 함수
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
        
        best_accuracy = 0.0
        
        print(f"학습 시작 - 에폭: {epochs}, 학습률: {lr}, 디바이스: {self.device}")
        
        for epoch in range(epochs):
            # 학습 모드
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            train_bar = tqdm(self.train_loader, desc=f"에폭 {epoch+1}/{epochs} [학습]")
            for inputs, targets in train_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # 순전파
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 역전파
                loss.backward()
                optimizer.step()
                
                # 통계
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 진행 상황 업데이트
                train_bar.set_postfix({
                    'loss': running_loss / (train_bar.n + 1),
                    'acc': 100. * correct / total
                })
            
            # 에폭 학습 결과
            train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100. * correct / total
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # 평가 모드
            test_loss, test_accuracy = self.evaluate()
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)
            
            # 학습률 조정
            scheduler.step(test_loss)
            
            print(f"에폭 {epoch+1}/{epochs} - 학습 손실: {train_loss:.4f}, 학습 정확도: {train_accuracy:.2f}%, "
                  f"테스트 손실: {test_loss:.4f}, 테스트 정확도: {test_accuracy:.2f}%")
            
            # 최고 성능 모델 저장
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                print(f"새로운 최고 정확도: {best_accuracy:.2f}% - 모델 저장 중...")
                torch.save(self.model.state_dict(), save_path)
        
        print(f"학습 완료 - 최고 테스트 정확도: {best_accuracy:.2f}%")
        
        # 최고 성능 모델 로드
        self.model.load_state_dict(torch.load(save_path))
        self.model.to(self.device)
        
        return self.model
    
    def evaluate(self):
        """
        모델 평가 함수
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            test_bar = tqdm(self.test_loader, desc="[평가]")
            for inputs, targets in test_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 순전파
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 통계
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 진행 상황 업데이트
                test_bar.set_postfix({
                    'loss': running_loss / (test_bar.n + 1),
                    'acc': 100. * correct / total
                })
        
        test_loss = running_loss / len(self.test_loader)
        test_accuracy = 100. * correct / total
        
        return test_loss, test_accuracy

# 테스트 함수
def test_trainer():
    from models.baseline_model import SimpleConvNet
    from utils.dataset_manager import DatasetManager
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    train_loader, test_loader, _ = dataset_manager.get_cifar10_dataset(batch_size=64)
    
    # 모델 생성
    model = SimpleConvNet()
    
    # 학습기 생성
    trainer = ModelTrainer(model, train_loader, test_loader)
    
    # 모델 학습
    trained_model = trainer.train(epochs=5, lr=0.001, save_path='./trained_model.pth')
    
    # 최종 평가
    test_loss, test_accuracy = trainer.evaluate()
    print(f"최종 테스트 손실: {test_loss:.4f}, 최종 테스트 정확도: {test_accuracy:.2f}%")
    
    return trained_model

if __name__ == "__main__":
    test_trainer()

