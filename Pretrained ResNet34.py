import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===============================================================
# 1. 데이터셋 경로 및 전처리 설정
# ===============================================================
# your_root/ 아래에 반드시 ILSVRC2012_img_val/, ILSVRC2012_devkit_t12/ 가 있어야 함
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

# ===============================================================
# 2. ImageNet Validation Set 로드
# ===============================================================
val_dataset = torchvision.datasets.ImageNet(
    root=root_dir,
    split='val',
    transform=val_transform
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ===============================================================
# 3. Pretrained ResNet34 모델 로드
# ===============================================================
model = torchvision.models.resnet34(pretrained=True)
model.eval()
model.cuda()  # CUDA 사용 (가능한 경우)

# ===============================================================
# 4. 평가 함수 정의 (Top-1, Top-5 정확도)
# ===============================================================
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

# ===============================================================
# 5. 평가 실행
# ===============================================================
top1_acc, top5_acc = evaluate(model, val_loader)
print(f"\n✅ ResNet34 on ImageNet Val:")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")
