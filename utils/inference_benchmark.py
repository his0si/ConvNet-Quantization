import torch
import time
import numpy as np
from tqdm import tqdm

class InferenceBenchmark:
    """
    모델 추론 속도를 벤치마크하는 클래스
    """
    def __init__(self, test_loader):
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def warm_up(self, model, num_iterations=10):
        """
        모델 워밍업을 위한 함수 (캐시 효과 제거)
        """
        model.eval()
        model = model.to(self.device)
        
        # 실제 데이터로 워밍업
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                if i >= num_iterations:
                    break
                data = data.to(self.device)
                _ = model(data)
    
    def measure_inference_time(self, model, batch_size=1, num_iterations=100, verbose=True):
        """
        단일 이미지 및 배치 추론 시간을 측정하는 함수
        """
        model.eval()
        model = model.to(self.device)
        
        # 워밍업
        if verbose:
            print("모델 워밍업 중...")
        self.warm_up(model)
        
        # 실제 데이터로 추론 시간 측정
        single_times = []
        batch_times = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                if i >= num_iterations:
                    break
                
                # 단일 이미지 추론 시간 측정
                for j in range(data.size(0)):
                    single_data = data[j:j+1].to(self.device)
                    start_time = time.time()
                    _ = model(single_data)
                    end_time = time.time()
                    single_times.append((end_time - start_time) * 1000)  # ms 단위로 변환
                
                # 배치 추론 시간 측정
                batch_data = data.to(self.device)
                start_time = time.time()
                _ = model(batch_data)
                end_time = time.time()
                batch_times.append((end_time - start_time) * 1000)  # ms 단위로 변환
        
        # 결과 계산
        single_mean = np.mean(single_times)
        single_std = np.std(single_times)
        batch_mean = np.mean(batch_times)
        batch_std = np.std(batch_times)
        
        if verbose:
            print(f"단일 이미지 추론 시간: {single_mean:.2f} ± {single_std:.2f} ms")
            print(f"배치 크기 {batch_size}의 추론 시간: {batch_mean:.2f} ± {batch_std:.2f} ms")
            print(f"이미지당 평균 추론 시간 (배치 처리): {batch_mean / batch_size:.2f} ms")
        
        return {
            'single_mean': single_mean,
            'single_std': single_std,
            'batch_mean': batch_mean,
            'batch_std': batch_std,
            'per_image_in_batch': batch_mean / batch_size
        }
    
    def measure_throughput(self, model, batch_sizes=[1, 4, 8, 16, 32, 64], num_iterations=100, verbose=True):
        """
        다양한 배치 크기에서의 처리량(throughput)을 측정하는 함수
        """
        model.eval()
        model = model.to(self.device)
        
        # 워밍업
        if verbose:
            print("모델 워밍업 중...")
        self.warm_up(model)
        
        results = {}
        
        for batch_size in batch_sizes:
            if verbose:
                print(f"배치 크기 {batch_size}의 처리량 측정 중...")
            
            # 측정 시작
            start_time = time.time()
            total_images = 0
            
            with torch.no_grad():
                for i, (data, _) in enumerate(self.test_loader):
                    if i >= num_iterations:
                        break
                    data = data.to(self.device)
                    _ = model(data)
                    total_images += data.size(0)
            
            # 측정 종료
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 처리량 계산 (images/sec)
            throughput = total_images / elapsed_time
            
            if verbose:
                print(f"배치 크기 {batch_size}의 처리량: {throughput:.2f} images/sec")
            
            results[batch_size] = throughput
        
        return results
    
    def compare_models(self, models_dict, batch_size=32, verbose=True):
        """
        여러 모델의 추론 속도를 비교하는 함수
        """
        results = {}
        
        if verbose:
            print("모델 추론 속도 비교 시작...")
        
        for name, model in models_dict.items():
            if verbose:
                print(f"\n{name} 모델 벤치마크 중...")
            
            # 추론 시간 측정
            inference_times = self.measure_inference_time(model, batch_size=batch_size, verbose=verbose)
            
            # 처리량 측정
            throughput = self.measure_throughput(model, batch_sizes=[1, batch_size], verbose=verbose)
            
            results[name] = {
                'inference_times': inference_times,
                'throughput': throughput
            }
        
        # 결과 요약
        if verbose:
            print("\n=== 모델 추론 속도 비교 결과 ===")
            for name, result in results.items():
                print(f"{name}:")
                print(f"  단일 이미지 추론 시간: {result['inference_times']['single_mean']:.2f} ms")
                print(f"  배치 처리 시 이미지당 추론 시간: {result['inference_times']['per_image_in_batch']:.2f} ms")
                print(f"  처리량 (배치 크기 {batch_size}): {result['throughput'][batch_size]:.2f} images/sec")
        
        return results

# 테스트 함수
def test_benchmark():
    from utils.dataset_manager import DatasetManager
    import torchvision
    
    # 데이터셋 로드
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_imagenet_dataset()
    
    # 벤치마크 생성
    benchmark = InferenceBenchmark(test_loader)
    
    # 테스트용 모델 생성
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 추론 시간 측정
    print("모델 추론 시간 측정 테스트:")
    inference_times = benchmark.measure_inference_time(model, batch_size=32)
    
    # 처리량 측정
    print("\n모델 처리량 측정 테스트:")
    throughput = benchmark.measure_throughput(model, batch_sizes=[1, 8, 32])
    
    return benchmark

if __name__ == "__main__":
    test_benchmark()

