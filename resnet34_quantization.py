import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torchvision.models.quantization import resnet50 as quantized_resnet50
from torch.utils.data import DataLoader, random_split, Subset

import time
import copy
import os
import numpy as np

# --- 0. Configuration ---
print("--- Configuration ---")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quantization backend (CPU 전용 설정)
if DEVICE.type == 'cpu':
    SUPPORTED_ENGINES = torch.backends.quantized.supported_engines
    if 'fbgemm' in SUPPORTED_ENGINES:
        torch.backends.quantized.engine = 'fbgemm'
    elif 'qnnpack' in SUPPORTED_ENGINES:
        torch.backends.quantized.engine = 'qnnpack'
    else:
        print("Warning: Neither fbgemm nor qnnpack supported on CPU.")
        if SUPPORTED_ENGINES: torch.backends.quantized.engine = SUPPORTED_ENGINES[0]
        else: raise RuntimeError("Error: No quantization engine found.")
    print(f"Using CPU quantization engine: {torch.backends.quantized.engine}")
else:
    print(f"Running on {DEVICE}. Quantization behavior might differ from standard CPU comparison.")

# Dataset & Model Params
NUM_CLASSES = 10
INPUT_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 1
TRAIN_RATIO = 0.8
CALIBRATION_RATIO = 0.1

# Paths
MODEL_SAVE_DIR = "saved_models_cifar"
FP32_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "resnet50_cifar10_fp32_trained.pth")
STANDARD_PTQ_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "resnet50_cifar10_standard_ptq.pth")
CUSTOM_QUANT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "resnet50_cifar10_custom_quant.pth")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"Models will be saved in: {MODEL_SAVE_DIR}")
print(f"Using device: {DEVICE}")
print("-" * 30)

# --- 1. Data Loading and Preprocessing ---
print("--- Data Preparation (CIFAR-10) ---")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    normalize,
])

try:
    full_train_dataset_raw = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
except Exception as e:
    print(f"Error downloading/loading CIFAR10: {e}.")
    exit()

num_train_total = len(full_train_dataset_raw)
indices = list(range(num_train_total))
train_size = int(TRAIN_RATIO * num_train_total)
calib_size = int(CALIBRATION_RATIO * num_train_total)
test_size = num_train_total - train_size - calib_size

np.random.seed(42)
np.random.shuffle(indices)
train_indices = indices[:train_size]
calib_indices = indices[train_size:train_size + calib_size]
test_indices = indices[train_size + calib_size:]

class DatasetFromIndices(torch.utils.data.Dataset):
    def __init__(self, underlying_dataset, indices, transform=None):
        self.underlying_dataset = underlying_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample, label = self.underlying_dataset[real_idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

train_dataset = DatasetFromIndices(full_train_dataset_raw, train_indices, transform=transform_train)
calibration_dataset = DatasetFromIndices(full_train_dataset_raw, calib_indices, transform=transform_test)
test_dataset_from_train = DatasetFromIndices(full_train_dataset_raw, test_indices, transform=transform_test)

num_workers = 2 if DEVICE.type == 'cuda' else 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if DEVICE.type == 'cuda' else False)
calibration_loader = DataLoader(calibration_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset_from_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

print("DataLoaders created.")
print("-" * 30)

# --- 2. Helper Functions ---
def modify_resnet_fc(model, num_classes):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def fuse_resnet_modules(model):
    modules_to_fuse_names = []
    if isinstance(model, models.ResNet) and hasattr(model, 'conv1') and hasattr(model, 'bn1') and hasattr(model, 'relu'):
        modules_to_fuse_names.append(['conv1', 'bn1', 'relu'])
    for name, module in model.named_modules():
        if isinstance(module, models.resnet.Bottleneck):
            block_prefix = name
            if hasattr(module, 'conv1') and hasattr(module, 'bn1') and hasattr(module, 'relu'): modules_to_fuse_names.append([f'{block_prefix}.conv1', f'{block_prefix}.bn1', f'{block_prefix}.relu'])
            if hasattr(module, 'conv2') and hasattr(module, 'bn2') and hasattr(module, 'relu'): modules_to_fuse_names.append([f'{block_prefix}.conv2', f'{block_prefix}.bn2', f'{block_prefix}.relu'])
            if hasattr(module, 'conv3') and hasattr(module, 'bn3'): modules_to_fuse_names.append([f'{block_prefix}.conv3', f'{block_prefix}.bn3'])
            if module.downsample is not None and len(module.downsample) == 2 and isinstance(module.downsample[0], nn.Conv2d) and isinstance(module.downsample[1], nn.BatchNorm2d): modules_to_fuse_names.append([f'{block_prefix}.downsample.0', f'{block_prefix}.downsample.1'])
    if not modules_to_fuse_names: print("Warning: No standard ResNet module groups identified for fusion."); return model
    print(f"Identified {len(modules_to_fuse_names)} module groups for potential fusion.")
    fused_model = copy.deepcopy(model); fused_model.eval()
    try:
        valid_modules_to_fuse = []
        for group in modules_to_fuse_names:
            try: current_module = fused_model; [getattr(current_module := getattr(current_module, part), '') for part in group[0].split('.')]; valid_modules_to_fuse.append(group)
            except AttributeError: print(f"Warning: Module group path invalid, skipping fusion: {group}")
        if not valid_modules_to_fuse: print("Error: No valid modules found to fuse after path checking."); return model
        print(f"Attempting to fuse {len(valid_modules_to_fuse)} valid module groups..."); torch.quantization.fuse_modules(fused_model, valid_modules_to_fuse, inplace=True); print("Fusion successful."); return fused_model
    except Exception as e: print(f"Error during fusion: {e}"); return model

@torch.no_grad()
def evaluate_model(model, dataloader, device, description="Evaluation"):
    model.eval(); model.to(device); correct = 0; total = 0
    print(f"{description} running..."); loader_len = len(dataloader)
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        try: outputs = model(inputs); _, predicted = torch.max(outputs.data, 1); total += labels.size(0); correct += (predicted == labels).sum().item()
        except Exception as e: print(f"Error during {description} batch {i+1}/{loader_len}: {e}"); raise e
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"{description} finished."); return accuracy

@torch.no_grad()
def measure_inference_time(model, dummy_input, device, iterations=30):
    model.eval(); model.to(device)
    dummy_input_dev = dummy_input.to(device)
    try:
        for _ in range(10): _ = model(dummy_input_dev)
        if device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(iterations): _ = model(dummy_input_dev)
        if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.perf_counter()
        avg_time_ms = (end_time - start_time) / iterations * 1000
        return avg_time_ms
    except Exception as e:
        print(f"Error during inference time measurement: {e}")
        return float('nan')

def get_model_size(model, file_path, is_jit=False):
    try:
        if is_jit: torch.jit.save(model, file_path)
        else: torch.save(model.state_dict(), file_path)
        size_mb = os.path.getsize(file_path) / 1e6
        return size_mb
    except Exception as e: print(f"Error saving model to {file_path}: {e}"); return float('nan')

# --- 3. Model Training ---
print("\n--- Model Training (ResNet50 on CIFAR-10) ---")
fp32_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
fp32_model = modify_resnet_fc(fp32_model, NUM_CLASSES)
fp32_model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fp32_model.parameters(), lr=LEARNING_RATE)

if os.path.exists(FP32_MODEL_PATH) and NUM_EPOCHS == 0:
    print(f"Loading previously trained model from {FP32_MODEL_PATH}")
    fp32_model.load_state_dict(torch.load(FP32_MODEL_PATH, map_location=DEVICE))
else:
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    fp32_model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = fp32_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
    print("Training finished.")
    print(f"Saving trained model to {FP32_MODEL_PATH}")
    torch.save(fp32_model.state_dict(), FP32_MODEL_PATH)

print("-" * 30)

# --- 4. FP32 Baseline Evaluation ---
print("\n--- FP32 Baseline Evaluation ---")
fp32_model.eval()
fp32_accuracy = evaluate_model(fp32_model, test_loader, DEVICE, "FP32 Baseline Evaluation")
dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
fp32_inference_time = measure_inference_time(fp32_model, dummy_input, DEVICE)
fp32_model_size = get_model_size(fp32_model, FP32_MODEL_PATH.replace(".pth", "_eval.pth"), is_jit=False)

print(f"FP32 Baseline Accuracy: {fp32_accuracy:.2f}%")
print(f"FP32 Baseline Inference Time: {fp32_inference_time:.3f} ms")
print(f"FP32 Baseline Size: {fp32_model_size:.2f} MB")
print("-" * 30)

# --- 5. Standard PTQ ---
print("\n--- Standard PTQ ---")
ptq_device = torch.device("cpu")
ptq_accuracy, ptq_inference_time, ptq_model_size = float('nan'), float('nan'), float('nan')

try:
    fp32_model_cpu = models.resnet50(weights=None)
    fp32_model_cpu = modify_resnet_fc(fp32_model_cpu, NUM_CLASSES)
    fp32_model_cpu.load_state_dict(torch.load(FP32_MODEL_PATH, map_location=ptq_device))
    fp32_model_cpu.eval()

    fused_model_ptq = fuse_resnet_modules(fp32_model_cpu)
    qconfig_ptq = torch.quantization.get_default_qconfig('fbgemm' if 'fbgemm' in SUPPORTED_ENGINES else 'qnnpack')
    fused_model_ptq.qconfig = qconfig_ptq
    torch.quantization.prepare(fused_model_ptq, inplace=True)

    with torch.no_grad():
        for inputs, _ in calibration_loader:
            fused_model_ptq(inputs.to(ptq_device))

    quantized_model_ptq = torch.quantization.convert(fused_model_ptq, inplace=True)
    quantized_model_ptq.eval()

    ptq_accuracy = evaluate_model(quantized_model_ptq, test_loader, ptq_device, "Standard PTQ Evaluation")
    dummy_input_ptq = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(ptq_device)
    ptq_inference_time = measure_inference_time(quantized_model_ptq, dummy_input_ptq, ptq_device)
    ptq_model_size = get_model_size(quantized_model_ptq, STANDARD_PTQ_MODEL_PATH, is_jit=False)
except Exception as ptq_err:
    print(f"\n!!!! Error during Standard PTQ workflow: {ptq_err} !!!!")

print(f"Standard PTQ Accuracy: {ptq_accuracy:.2f}%")
print(f"Standard PTQ Inference Time: {ptq_inference_time:.3f} ms")
print(f"Standard PTQ Size: {ptq_model_size:.2f} MB")
print("-" * 30)

# --- 6. Custom Quantization (Quantized ResNet50 Eager Mode on CPU) ---
print("\n--- Custom Quantization (Quantized ResNet50 Eager Mode on CPU) ---")
custom_q_device = torch.device("cpu") # CPU에서 수행
custom_accuracy, custom_inference_time, custom_model_size = float('nan'), float('nan'), float('nan')

try:
    # 6.1 Initialize Quantized Model Stub
    # pretrained=False 또는 weights=None 으로 초기화 후 학습된 가중치 로드
    custom_quant_model = quantized_resnet50(weights=None, quantize=False)
    custom_quant_model = modify_resnet_fc(custom_quant_model, NUM_CLASSES)

    # 6.2 Load Trained FP32 Weights
    print("Loading trained FP32 weights into Quantized ResNet50 stub...")
    custom_quant_model.load_state_dict(torch.load(FP32_MODEL_PATH, map_location=custom_q_device), strict=False)
    custom_quant_model.to(custom_q_device)
    custom_quant_model.eval()
    print("Weight loading successful.")

    # 6.3 Fuse
    print("Fusing Quantized ResNet50 model...")
    custom_quant_fused_model = fuse_resnet_modules(custom_quant_model)

    # 6.4 Prepare Eager
    print("Preparing Quantized ResNet50 (Eager) for PTQ...")
    qconfig_custom = torch.quantization.get_default_qconfig(torch.backends.quantized.engine)
    custom_quant_fused_model.qconfig = qconfig_custom
    torch.quantization.prepare(custom_quant_fused_model, inplace=True)
    print("Preparation done.")

    # 6.5 Calibrate Eager
    print("Calibrating Quantized ResNet50 (Eager)...")
    with torch.no_grad():
        for inputs, _ in calibration_loader: custom_quant_fused_model(inputs.to(custom_q_device))
    print("Calibration done.")

    # 6.6 Convert Eager
    print("Converting Quantized ResNet50 (Eager)...")
    custom_quantized_model_final = torch.quantization.convert(custom_quant_fused_model, inplace=True)
    custom_quantized_model_final.eval()
    print("Conversion successful.")

    # 6.7 Evaluate Eager Model
    custom_accuracy = evaluate_model(custom_quantized_model_final, test_loader, custom_q_device, "Custom Quant (Eager) Evaluation")
    dummy_input_custom = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(custom_q_device)
    custom_inference_time = measure_inference_time(custom_quantized_model_final, dummy_input_custom, custom_q_device)
    custom_model_size = get_model_size(custom_quantized_model_final, CUSTOM_QUANT_MODEL_PATH, is_jit=False)

except Exception as custom_err:
    print(f"\n!!!! Error during Custom Quantization (Eager) workflow: {custom_err} !!!!")

print(f"Custom Quant Accuracy: {custom_accuracy:.2f}%")
print(f"Custom Quant Inference Time: {custom_inference_time:.3f} ms")
print(f"Custom Quant Size: {custom_model_size:.2f} MB")
print("-" * 30)

# --- 7. Final Comparison Summary ---
print("\n--- Final Comparison Summary (ResNet50 on CIFAR-10) ---")
print(f"(Dataset: CIFAR10, Train {TRAIN_RATIO*100}%, Calib {CALIBRATION_RATIO*100}%, Test {100-TRAIN_RATIO*100-CALIBRATION_RATIO*100}% from Train Split)")
print(f"(Trained for {NUM_EPOCHS} epochs)")

header = f"| {'Metric':<18} | {'FP32 Baseline':<15} | {'Standard PTQ (JIT)':<20} | {'Custom Quant (Eager)':<22} |"
separator = "-" * len(header)
print(separator); print(header); print(separator)

def format_value(value, precision, width=15):
    if isinstance(value, str): return f"{value:<{width}}"
    if value is None or np.isnan(value): return f"{'Error':<{width}}"
    if isinstance(value, (float, int)): return f"{value:<{width}.{precision}f}"
    return f"{str(value):<{width}}"
def format_diff(value, base, precision, width=15):
    if isinstance(value, str) or isinstance(base, str) or value is None or np.isnan(value) or base is None or np.isnan(base): return f"{'Error':<{width}}"
    diff = value - base; return f"{diff:<+{width}.{precision}f}"
def format_speedup(value, base, width=15):
    if isinstance(value, str) or isinstance(base, str) or value is None or np.isnan(value) or value == 0 or base is None or np.isnan(base): return f"{'Error':<{width}}"
    speedup = base / value; return f"{speedup:<{width}.2f}"

ptq_acc_f = format_value(ptq_accuracy, 2, width=20)
ptq_time_f = format_value(ptq_inference_time, 3, width=20)
ptq_size_f = format_value(ptq_model_size, 2, width=20)
ptq_acc_diff_f = format_diff(ptq_accuracy, fp32_accuracy, 2, width=20)
ptq_speedup_f = format_speedup(ptq_inference_time, fp32_inference_time, width=20)

custom_acc_f = format_value(custom_accuracy, 2, width=22)
custom_time_f = format_value(custom_inference_time, 3, width=22)
custom_size_f = format_value(custom_model_size, 2, width=22)
custom_acc_diff_f = format_diff(custom_accuracy, fp32_accuracy, 2, width=22)
custom_speedup_f = format_speedup(custom_inference_time, fp32_inference_time, width=22)

print(f"| {'Accuracy (%)':<18} | {format_value(fp32_accuracy, 2):<15} | {ptq_acc_f} | {custom_acc_f} |")
print(f"| {'Acc. Drop (%)':<18} | {'-':<15} | {ptq_acc_diff_f} | {custom_acc_diff_f} |")
print(f"| {'Inference (ms)':<18} | {format_value(fp32_inference_time, 3):<15} | {ptq_time_f} | {custom_time_f} |")
print(f"| {'Speedup (x)':<18} | {'1.00':<15} | {ptq_speedup_f} | {custom_speedup_f} |")
print(f"| {'Size (MB)':<18} | {format_value(fp32_model_size, 2):<15} | {ptq_size_f} | {custom_size_f} |")
print(separator)

print("\nNotes:")
print(f"- Training performed for {NUM_EPOCHS} epochs. For better accuracy, increase epochs significantly.")
print("- Accuracy measured on a 10% split of the CIFAR-10 *training* set.")
print("- Standard PTQ uses TorchScript JIT approach on CPU due to prior eager mode errors.")
print("- Custom Quant uses torchvision's quantized_resnet50 with eager mode PTQ on CPU.")
print(f"- All quantization steps (calibration, conversion, evaluation) were performed on CPU ({torch.backends.quantized.engine} backend).")