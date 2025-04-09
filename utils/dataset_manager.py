import torch
import torchvision
import torchvision.transforms as transforms
import os

class DatasetManager:
    """
    평가용 데이터셋을 생성하고 관리하는 클래스
    """
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.train_loader = None
        self.test_loader = None
        self.calibration_loader = None
        
        # 데이터셋 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
    
    def get_cifar10_dataset(self, batch_size=128, num_workers=2):
        """
        CIFAR-10 데이터셋을 로드하고 데이터 로더를 생성하는 함수
        """
        # 데이터 전처리 및 증강
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # 학습 데이터셋 로드
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=transform_train
        )
        
        # 테스트 데이터셋 로드
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=transform_test
        )
        
        # 데이터 로더 생성
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        # 캘리브레이션용 데이터 로더 (학습 데이터의 일부만 사용)
        calibration_dataset = torch.utils.data.Subset(
            train_dataset, 
            indices=range(1000)  # 1000개 샘플만 사용
        )
        
        self.calibration_loader = torch.utils.data.DataLoader(
            calibration_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        print(f"학습 데이터셋 크기: {len(train_dataset)}")
        print(f"테스트 데이터셋 크기: {len(test_dataset)}")
        print(f"캘리브레이션 데이터셋 크기: {len(calibration_dataset)}")
        
        # 클래스 이름
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                         'dog', 'frog', 'horse', 'ship', 'truck')
        
        return self.train_loader, self.test_loader, self.calibration_loader
    
    def get_sample_batch(self):
        """
        시각화 및 테스트를 위한 샘플 배치를 반환하는 함수
        """
        if self.test_loader is None:
            self.get_cifar10_dataset()
        
        # 테스트 데이터에서 첫 번째 배치 가져오기
        data_iter = iter(self.test_loader)
        images, labels = next(data_iter)
        
        return images, labels, self.classes

# 테스트 함수
def test_dataset():
    dataset_manager = DatasetManager()
    train_loader, test_loader, calibration_loader = dataset_manager.get_cifar10_dataset()
    
    # 샘플 배치 가져오기
    images, labels, classes = dataset_manager.get_sample_batch()
    
    print(f"샘플 배치 이미지 형태: {images.shape}")
    print(f"샘플 배치 레이블 형태: {labels.shape}")
    
    # 레이블 확인
    print("샘플 이미지 클래스:")
    for i in range(min(5, len(labels))):
        print(f"이미지 {i}: {classes[labels[i]]}")
    
    return dataset_manager

if __name__ == "__main__":
    test_dataset()

