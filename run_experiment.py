import torch
import os
import argparse
from models.baseline_model import SimpleConvNet
from utils.dataset_manager import DatasetManager
from model_trainer import ModelTrainer
from compare_quantization import apply_quantization_and_compare

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='모델 학습 및 양자화 비교 실행')
    parser.add_argument('--epochs', type=int, default=20, help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=128, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='모델 저장 디렉토리')
    parser.add_argument('--skip_training', action='store_true', help='학습 과정 건너뛰기')
    args = parser.parse_args()
    
    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'trained_fp32_model.pth')
    
    # 1. 모델 학습 (필요한 경우)
    if not args.skip_training or not os.path.exists(save_path):
        print("=" * 50)
        print("1단계: FP32 모델 학습")
        print("=" * 50)
        
        # 데이터셋 로드
        print("데이터셋 로드 중...")
        dataset_manager = DatasetManager()
        train_loader, test_loader, _ = dataset_manager.get_cifar10_dataset(batch_size=args.batch_size)
        
        # 모델 생성
        print("FP32 모델 생성 중...")
        model = SimpleConvNet()
        
        # 디바이스 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"학습 디바이스: {device}")
        
        # 학습기 생성
        trainer = ModelTrainer(model, train_loader, test_loader, device)
        
        # 모델 학습
        print(f"모델 학습 시작 (에폭: {args.epochs}, 학습률: {args.lr})...")
        trained_model = trainer.train(epochs=args.epochs, lr=args.lr, save_path=save_path)
        
        # 최종 평가
        print("최종 모델 평가 중...")
        test_loss, test_accuracy = trainer.evaluate()
        print(f"최종 테스트 손실: {test_loss:.4f}, 최종 테스트 정확도: {test_accuracy:.2f}%")
        
        print(f"학습된 모델이 {save_path}에 저장되었습니다.")
    else:
        print(f"학습 과정을 건너뛰고 저장된 모델을 사용합니다: {save_path}")
    
    # 2. 양자화 적용 및 비교
    print("\n" + "=" * 50)
    print("2단계: 양자화 적용 및 성능 비교")
    print("=" * 50)
    
    # 양자화 적용 및 비교
    models, results = apply_quantization_and_compare(save_path)
    
    print("\n" + "=" * 50)
    print("모든 과정이 완료되었습니다!")
    print("결과 그래프는 ./results 디렉토리에 저장되었습니다.")
    print("=" * 50)

if __name__ == "__main__":
    main()

