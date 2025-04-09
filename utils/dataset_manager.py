import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets

class DatasetManager:
    """
    평가용 데이터셋을 생성하고 관리하는 클래스
    """
    def __init__(self, data_dir='./imagenet'):
        self.data_dir = data_dir
        self.test_loader = None
        self.classes = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 데이터셋 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        
        # 데이터 전처리 설정
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_sample_batch(self):
        """
        시각화 및 테스트를 위한 샘플 배치를 반환하는 함수
        """
        if self.test_loader is None:
            self.get_imagenet_dataset()
        
        try:
            # 테스트 데이터에서 첫 번째 배치 가져오기
            data_iter = iter(self.test_loader)
            images, labels = next(data_iter)
            return images, labels
        except Exception as e:
            print(f"Error getting sample batch: {e}")
            raise

    def get_imagenet_dataset(self, batch_size=32, num_workers=4):
        """
        ImageNet 데이터셋을 로드하고 데이터 로더를 생성하는 함수
        """
        try:
            # 테스트 데이터셋 로드
            test_dataset = torchvision.datasets.ImageNet(
                root=self.data_dir,
                split='val',
                transform=self.transform
            )

            # Set the classes attribute
            self.classes = test_dataset.classes

            # 데이터 로더 생성
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            print(f"테스트 데이터셋 크기: {len(test_dataset)}")
            return self.test_loader
        except Exception as e:
            print(f"Error in get_imagenet_dataset: {e}")
            raise

    def evaluate_model(self, model):
        """
        모델의 Top-1, Top-5 정확도를 평가하는 함수
        """
        model = model.to(self.device)
        model.eval()
        
        correct1 = 0
        correct5 = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(images)
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                
                correct1 += correct[:1].view(-1).float().sum(0).item()
                correct5 += correct[:5].view(-1).float().sum(0).item()
                total += targets.size(0)
        
        top1 = 100.0 * correct1 / total
        top5 = 100.0 * correct5 / total
        
        return top1, top5

    def get_cifar10_dataset(self, batch_size=64):
        """
        CIFAR-10 데이터셋을 로드하는 함수
        """
        # CIFAR-10 데이터셋 로드
        train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.cifar10_transform
        )
        
        test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.cifar10_transform
        )
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, test_loader, test_dataset.classes

def test_dataset():
    try:
        dataset_manager = DatasetManager()
        test_loader = dataset_manager.get_imagenet_dataset()
        
        # 샘플 배치 가져오기
        images, labels = dataset_manager.get_sample_batch()
        
        print(f"샘플 배치 이미지 형태: {images.shape}")
        print(f"샘플 배치 레이블 형태: {labels.shape}")
        
        return dataset_manager
    except Exception as e:
        print(f"Error in test_dataset: {e}")
        raise

if __name__ == "__main__":
    test_dataset()

# 데이터셋 경로 및 전처리 설정
root_dir = './imagenet'

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ImageNet Validation Set 로드
val_dataset = torchvision.datasets.ImageNet(
    root=root_dir,
    split='val',
    transform=val_transform
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Pretrained ResNet50 모델 로드
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()
model.cuda()  # CUDA 사용 (가능한 경우)

# 평가 함수 정의 (Top-1, Top-5 정확도)
def evaluate(model, dataloader):
    correct1, correct5, total = 0, 0, 0
    with torch.inference_mode():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.cuda()
            targets = targets.cuda()
            outputs = model(images)
            _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct1 += (pred_top5[:, 0] == targets).sum().item()
            correct5 += (pred_top5 == targets.unsqueeze(1)).sum().item()
            total += targets.size(0)
    top1 = correct1 / total * 100
    top5 = correct5 / total * 100
    return top1, top5

# 평가 실행
top1_acc, top5_acc = evaluate(model, val_loader)
print(f"\n✅ ResNet50 on ImageNet Val:")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")

