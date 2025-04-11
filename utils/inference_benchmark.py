import torch
import time
import numpy as np
from tqdm import tqdm

class InferenceBenchmark:
    """
    모델 추론 속도를 벤치마크하는 클래스
    """
    def __init__(self, test_loader, device='cpu'):
        self.test_loader = test_loader
        self.device = device
    
    def warm_up(self, model, num_iterations=10):
        """
        모델 워밍업을 위한 함수 (캐시 효과 제거)
        """
        print("모델 워밍업 중...")
        model.eval()
        model.to(self.device)
        
        # 테스트 데이터 로더에서 배치 가져오기
        data, _ = next(iter(self.test_loader))
        data = data.to(self.device)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(data)
    
    def measure_inference_time(self, model, batch_size=1, num_iterations=100, verbose=True):
        """
        단일 이미지 및 배치 추론 시간을 측정하는 함수
        """
        model.eval()
        model.to(self.device)
        
        # 단일 이미지 추론 시간 측정
        data, _ = next(iter(self.test_loader))
        data = data[0].unsqueeze(0).to(self.device)  # 단일 이미지
        
        single_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(data)
                end_time = time.time()
                single_times.append((end_time - start_time) * 1000)  # ms 단위로 변환
        
        single_mean = np.mean(single_times)
        single_std = np.std(single_times)
        
        if verbose:
            print(f"단일 이미지 추론 시간: {single_mean:.2f} ± {single_std:.2f} ms")
        
        # 배치 처리 시간 측정
        batch_data, _ = next(iter(self.test_loader))
        batch_data = batch_data[:batch_size].to(self.device)
        
        batch_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(batch_data)
                end_time = time.time()
                batch_times.append((end_time - start_time) * 1000)  # ms 단위로 변환
        
        batch_mean = np.mean(batch_times)
        batch_std = np.std(batch_times)
        per_image_mean = batch_mean / batch_size
        
        if verbose:
            print(f"배치 크기 {batch_size}의 추론 시간: {batch_mean:.2f} ± {batch_std:.2f} ms")
            print(f"이미지당 평균 추론 시간 (배치 처리): {per_image_mean:.2f} ms")
        
        return {
            'single': (single_mean, single_std),
            'batch': (batch_mean, batch_std),
            'per_image': per_image_mean
        }
    
    def measure_throughput(self, model, batch_size=1, num_iterations=100, verbose=True):
        """
        다양한 배치 크기에서의 처리량(throughput)을 측정하는 함수
        """
        model.eval()
        model.to(self.device)
        
        # 테스트 데이터 로더에서 배치 가져오기
        data, _ = next(iter(self.test_loader))
        data = data[:batch_size].to(self.device)
        
        total_time = 0
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(data)
                end_time = time.time()
                total_time += (end_time - start_time)
        
        throughput = (batch_size * num_iterations) / total_time
        
        if verbose:
            print(f"배치 크기 {batch_size}의 처리량: {throughput:.2f} images/sec")
        
        return throughput
    
    def compare_models(self, models_dict, batch_size=32, num_iterations=100, verbose=True):
        """
        여러 모델의 추론 속도를 비교하는 함수
        """
        print("\n모델 추론 속도 비교 시작...")
        
        results = {}
        for name, model in models_dict.items():
            print(f"\n{name} 모델 벤치마크 중...")
            
            # 워밍업
            self.warm_up(model)
            
            # 단일 이미지 및 배치 처리 시간 측정
            inference_times = self.measure_inference_time(
                model, batch_size=batch_size, 
                num_iterations=num_iterations, verbose=verbose
            )
            
            # 처리량 측정
            self.warm_up(model)
            print("배치 크기 1의 처리량 측정 중...")
            throughput_1 = self.measure_throughput(
                model, batch_size=1,
                num_iterations=num_iterations, verbose=verbose
            )
            
            print("배치 크기 32의 처리량 측정 중...")
            throughput_32 = self.measure_throughput(
                model, batch_size=32,
                num_iterations=num_iterations, verbose=verbose
            )
            
            results[name] = {
                'single_inference_time': inference_times['single'][0],
                'batch_inference_time': inference_times['batch'][0],
                'per_image_time': inference_times['per_image'],
                'throughput_1': throughput_1,
                'throughput_32': throughput_32
            }
        
        # 결과 출력
        print("\n=== 모델 추론 속도 비교 결과 ===")
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  단일 이미지 추론 시간: {result['single_inference_time']:.2f} ms")
            print(f"  배치 처리 시 이미지당 추론 시간: {result['per_image_time']:.2f} ms")
            print(f"  처리량 (배치 크기 32): {result['throughput_32']:.2f} images/sec")
        
        # 처리량만 반환 (결과 분석용)
        return {name: result['throughput_32'] for name, result in results.items()}

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
    throughput = benchmark.measure_throughput(model, batch_size=32)
    
    return benchmark

if __name__ == "__main__":
    test_benchmark()

