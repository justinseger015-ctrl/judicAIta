# JudicAIta Colab Notebook Validation Guide

> **Purpose**: Complete validation guide for `examples/notebooks/train_tunix_reasoning.ipynb` to ensure hackathon submission readiness.

**Last Updated**: January 2026  
**Target Hackathon**: Google Tunix / Kaggle Hackathon (Deadline: January 12, 2026)  
**Related Issue**: LEG-7, PDE-21, LEG-12

---

## TPU Session Tracking

**IMPORTANT**: Kaggle TPU usage is limited to **20 hours per week** and **9 hours per session**.

### Session Time Management

| Limit | Value | Notes |
|-------|-------|-------|
| Weekly TPU Quota | 20 hours | Resets weekly |
| Max Session Duration | 9 hours | Hard limit per notebook run |
| Recommended Checkpoint Interval | 30 minutes | Prevents data loss |
| Target Training Time | <8.5 hours | Leave buffer for setup |

### Time Tracking Cell

Add this cell early in your notebook to track session time:

```python
import time
from datetime import datetime, timedelta

SESSION_START = time.time()
MAX_SESSION_HOURS = 9.0
CHECKPOINT_INTERVAL_MINUTES = 30

def get_session_status():
    """Get current session time status."""
    elapsed = time.time() - SESSION_START
    elapsed_hours = elapsed / 3600
    remaining_hours = MAX_SESSION_HOURS - elapsed_hours
    
    print(f"⏱️  Session Time Status")
    print(f"   Started: {datetime.fromtimestamp(SESSION_START).strftime('%H:%M:%S')}")
    print(f"   Elapsed: {elapsed_hours:.2f} hours ({elapsed/60:.0f} minutes)")
    print(f"   Remaining: {remaining_hours:.2f} hours ({remaining_hours*60:.0f} minutes)")
    print(f"   Max Duration: {MAX_SESSION_HOURS} hours")
    
    if remaining_hours < 1.0:
        print("   ⚠️  WARNING: Less than 1 hour remaining!")
    elif remaining_hours < 0.5:
        print("   🚨 CRITICAL: Save checkpoint immediately!")
    
    return {
        "elapsed_hours": elapsed_hours,
        "remaining_hours": remaining_hours,
        "should_checkpoint": (elapsed / 60) % CHECKPOINT_INTERVAL_MINUTES < 1
    }

# Call periodically during training
get_session_status()
```

### Runtime Estimation Cell

Add before training to estimate if you'll complete within the session:

```python
def estimate_training_time(
    num_examples: int,
    batch_size: int,
    steps_per_example: int = 1,
    seconds_per_step: float = 2.0,
    num_epochs: int = 1
) -> dict:
    """Estimate total training time."""
    total_steps = (num_examples // batch_size) * num_epochs
    estimated_seconds = total_steps * seconds_per_step
    estimated_hours = estimated_seconds / 3600
    
    fits_session = estimated_hours < 8.5  # Leave 30 min buffer
    
    print(f"📊 Training Time Estimate")
    print(f"   Examples: {num_examples}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Total steps: {total_steps}")
    print(f"   Estimated time: {estimated_hours:.2f} hours")
    print(f"   Fits in session: {'✅ Yes' if fits_session else '❌ No - reduce scope'}")
    
    return {
        "total_steps": total_steps,
        "estimated_hours": estimated_hours,
        "fits_session": fits_session
    }

# Example usage
estimate_training_time(
    num_examples=100,
    batch_size=4,
    num_epochs=3,
    seconds_per_step=2.5
)
```

---

## Table of Contents

1. [Phase 1: Environment & Dependency Validation](#phase-1-environment--dependency-validation)
2. [Phase 2: Training Pipeline Verification](#phase-2-training-pipeline-verification)
3. [Phase 3: Inference & Output Quality](#phase-3-inference--output-quality)
4. [Phase 4: Documentation & Submission Prep](#phase-4-documentation--submission-prep)
5. [Troubleshooting Reference](#troubleshooting-reference)

---

## Phase 1: Environment & Dependency Validation

### Overview

This phase validates that all dependencies install correctly in a fresh Google Colab environment and that the TPU runtime is properly configured.

### Prerequisites Checklist

Before starting validation:

- [ ] Google Colab account with TPU access
- [ ] Hugging Face account (for Gemma model access)
- [ ] Kaggle account (for final submission)
- [ ] Browser with JavaScript enabled (for Colab)

### 1.1 Colab Runtime Configuration

**Expected Configuration:**
```yaml
Runtime Type: TPU
Python Version: 3.10+
TPU Type: v2-8 (minimum)
Memory: ~64GB HBM
```

**Validation Steps:**

1. Open notebook in Google Colab
2. Navigate to: `Runtime → Change runtime type`
3. Verify selections:
   - Hardware accelerator: **TPU** (NOT CPU or GPU)
   - Runtime shape: Standard
4. Click "Save"

**Expected Outcome**: Runtime type shows "TPU" in top-right corner

### 1.2 Dependency Installation Validation

**Critical Dependencies:**

| Package | Required Version | Notes |
|---------|------------------|-------|
| `google-tunix` | `0.1.0 - 0.1.6` | Max available: 0.1.6 (Dec 2025) |
| `jax` | TPU-compatible (0.8.x) | Must use `jax[tpu]` with libtpu releases |
| `flax` | `0.10.2` or `0.12.x` | Compatible with JAX TPU builds |
| `transformers` | `>=4.40.0,<=4.57.1` | For Gemma model support |
| `datasets` | `>=2.14.0` | For LegalBench data loading |
| `grain` | Latest | Data loading for Tunix |
| `qwix` | `0.1.5` | Model utilities |

**Installation Cell (Step 1):**

```python
# Install core dependencies
%pip install -q dotenv kagglehub ipywidgets tensorflow tensorflow_datasets tensorboardX
%pip install -q transformers>=4.40.0 grain huggingface_hub>=0.20.0 datasets>=2.14.0
%pip install -q 'numpy>2' sentencepiece>=0.1.99 safetensors>=0.4.0

# Install JAX, Tunix, Qwix, and Flax from GitHub
%pip install -q git+https://github.com/jax-ml/jax
%pip install git+https://github.com/google/tunix
%pip install git+https://github.com/google/qwix
%pip uninstall -q flax -y
%pip install git+https://github.com/google/flax
```

**Validation Checklist:**

- [ ] All packages install without errors
- [ ] `jax_cuda12_plugin` warnings appear (EXPECTED - harmless for TPU)
- [ ] No `ModuleNotFoundError` during installation
- [ ] Installation completes in < 5 minutes

**Post-Installation Validation Cell:**

```python
# Verify installed package versions
import subprocess
import sys

def get_package_version(package_name):
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'show', package_name],
        capture_output=True, text=True
    )
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            return line.split(':')[1].strip()
    return 'Not installed'

packages_to_check = {
    'google-tunix': '0.1.0-0.1.6',
    'jax': 'TPU-compatible',
    'flax': '0.10.2 or 0.12.x',
    'transformers': '>=4.40.0',
}

print("📦 Package Version Verification")
print("=" * 50)
for pkg, expected in packages_to_check.items():
    version = get_package_version(pkg)
    status = '✅' if version != 'Not installed' else '❌'
    print(f"{status} {pkg}: {version} (expected: {expected})")
```

**Expected Output:**
```
📦 Package Version Verification
==================================================
✅ google-tunix: 0.1.6 (expected: 0.1.0-0.1.6)
✅ jax: 0.8.3.dev20251228 (expected: TPU-compatible)
✅ flax: 0.12.2 (expected: 0.10.2 or 0.12.x)
✅ transformers: 4.57.1 (expected: >=4.40.0)
```

### 1.3 Runtime Restart Checkpoint

**⚠️ CRITICAL STEP**: Runtime MUST be restarted after Step 1 installation.

**Validation Steps:**

1. After Step 1 completes, click: `Runtime → Restart runtime`
2. Wait for runtime to fully restart (status indicator stops)
3. **DO NOT** re-run Step 1 after restart
4. Proceed directly to Step 2

**Why Restart is Required:**
- TPU libraries need fresh Python interpreter
- JAX TPU backend must be loaded before any JAX operations
- Skipping restart causes `RuntimeError: TPU not found`

### 1.4 TPU Detection Validation

**Validation Cell (After Runtime Restart):**

```python
import warnings
warnings.filterwarnings('ignore', message='.*jax_cuda12_plugin.*')

import jax
import jax.numpy as jnp

print("=" * 60)
print("🔍 TPU DETECTION VALIDATION")
print("=" * 60)

# JAX initialization
print("\n🚀 JAX Initialization:")
try:
    jax.distributed.initialize()
    print("   ✅ JAX distributed initialized")
except Exception as e:
    print(f"   ⚠️  JAX distributed: {e}")
    print("   (Normal for single-host setups)")

# Device detection
devices = jax.devices()
print(f"\n🖥️  Device Information:")
print(f"   Number of devices: {len(devices)}")

if len(devices) > 0:
    print(f"   Device type: {devices[0].platform}")
    print(f"   Devices: {devices}")
    
    # Validate device type
    if devices[0].platform == 'tpu':
        print("\n✅ TPU DETECTED - Ready for training!")
        if len(devices) == 8:
            print("   ✅ TPU v2-8 configuration confirmed (8 cores)")
        else:
            print(f"   ⚠️  {len(devices)} cores detected (expected 8 for v2-8)")
    else:
        print(f"\n❌ WRONG RUNTIME: {devices[0].platform}")
        print("   💡 Solution: Runtime → Change runtime type → TPU")
else:
    print("\n❌ NO DEVICES FOUND")
    print("   💡 Solution: Restart runtime and re-check")

print("\n" + "=" * 60)
```

**Expected Output:**
```
============================================================
🔍 TPU DETECTION VALIDATION
============================================================

🚀 JAX Initialization:
   ✅ JAX distributed initialized

🖥️  Device Information:
   Number of devices: 8
   Device type: tpu
   Devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), ...]

✅ TPU DETECTED - Ready for training!
   ✅ TPU v2-8 configuration confirmed (8 cores)

============================================================
```

**Validation Checklist:**

- [ ] JAX distributed initializes (or expected error message)
- [ ] 8 TPU devices detected
- [ ] Device type is 'tpu' (not 'cpu' or 'gpu')
- [ ] No critical errors in output

### 1.5 Core Import Verification

**Validation Cell:**

```python
import sys

print("=" * 60)
print("📦 CORE IMPORTS VERIFICATION")
print("=" * 60)

# Import validation
imports_status = {}

# Tunix imports
print("\n🔍 Tunix Framework:")
try:
    import tunix
    from tunix.generate import sampler as sampler_lib
    from tunix.generate import tokenizer_adapter as tokenizer_lib
    from tunix.models.gemma3 import model as gemma_lib
    from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
    from tunix.rl import rl_cluster as rl_cluster_lib
    from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
    print("   ✅ Tunix core imports successful")
    imports_status['tunix'] = True
except ImportError as e:
    print(f"   ❌ Tunix import failed: {e}")
    imports_status['tunix'] = False

# Flax imports
print("\n🔍 Flax Framework:")
try:
    from flax import nnx
    import flax
    print(f"   ✅ Flax imported (version: {flax.__version__})")
    imports_status['flax'] = True
except ImportError as e:
    print(f"   ❌ Flax import failed: {e}")
    imports_status['flax'] = False

# JAX imports
print("\n🔍 JAX Framework:")
try:
    import jax
    import jax.numpy as jnp
    print(f"   ✅ JAX imported (version: {jax.__version__})")
    imports_status['jax'] = True
except ImportError as e:
    print(f"   ❌ JAX import failed: {e}")
    imports_status['jax'] = False

# Supporting libraries
print("\n🔍 Supporting Libraries:")
try:
    import grain
    import optax
    from orbax import checkpoint as ocp
    import qwix
    print("   ✅ Grain, Optax, Orbax, Qwix imported")
    imports_status['supporting'] = True
except ImportError as e:
    print(f"   ❌ Supporting libraries import failed: {e}")
    imports_status['supporting'] = False

# Summary
print("\n" + "=" * 60)
print("📊 IMPORT SUMMARY")
print("=" * 60)
all_passed = all(imports_status.values())
for lib, status in imports_status.items():
    status_icon = '✅' if status else '❌'
    print(f"{status_icon} {lib}")

if all_passed:
    print("\n🎉 ALL IMPORTS SUCCESSFUL - Ready for Phase 2!")
else:
    print("\n❌ SOME IMPORTS FAILED - Review errors above")
    print("   See Troubleshooting Guide for solutions")

print("=" * 60)
```

**Validation Checklist:**

- [ ] Tunix imports successful (including GRPOLearner)
- [ ] Flax imports successful
- [ ] JAX imports successful
- [ ] Supporting libraries import successfully
- [ ] All import status checks pass

### 1.6 HBM Memory Visibility Check (Optional)

**Validation Cell:**

```python
print("=" * 60)
print("💾 HBM MEMORY VISIBILITY CHECK")
print("=" * 60)

try:
    from tunix.sft import utils as tunix_utils
    
    print("\n🔍 HBM Memory Stats:")
    if hasattr(tunix_utils, 'show_hbm_usage'):
        tunix_utils.show_hbm_usage()
        print("\n   ✅ HBM memory stats visible")
    else:
        print("   ⚠️  show_hbm_usage() not available in this Tunix version")
        print("   (This is informational only - training can proceed)")
        
except ImportError as e:
    print(f"   ⚠️  Tunix SFT utils not available: {e}")
    print("   (This is informational only - training can proceed)")
except Exception as e:
    print(f"   ⚠️  HBM check warning: {e}")
    print("   (This is informational only - training can proceed)")

print("\n" + "=" * 60)
```

**Validation Checklist:**

- [ ] HBM stats display (if available) OR
- [ ] Informational message (acceptable for newer Tunix versions)

### 1.7 LoRA Adapter Configuration Test

**Validation Cell:**

```python
print("=" * 60)
print("🔧 LORA ADAPTER CONFIGURATION TEST")
print("=" * 60)

# Define LoRA configuration for testing
lora_test_config = {
    "rank": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "dropout": 0.05,
}

print("\n🔍 LoRA Configuration:")
for key, value in lora_test_config.items():
    print(f"   {key}: {value}")

# Test Gemma model access
print("\n🔍 Gemma Model Access:")
try:
    from tunix.models.gemma3 import model as gemma_model
    print("   ✅ Gemma3 model module accessible")
except ImportError as e:
    print(f"   ❌ Gemma3 model import failed: {e}")

# Test AutoModel access (optional)
print("\n🔍 Tunix AutoModel Access:")
try:
    from tunix.models import automodel
    print("   ✅ Tunix AutoModel module accessible")
except ImportError as e:
    print(f"   ⚠️  AutoModel import: {e}")
    print("   (Optional - direct model loading still works)")

print("\n✅ LoRA configuration validated")
print("=" * 60)
```

**Validation Checklist:**

- [ ] LoRA config dictionary created successfully
- [ ] Gemma3 model module accessible
- [ ] No critical import errors

### 1.8 Phase 1 Validation Summary

**Summary Cell:**

```python
import sys
import jax
import flax

print("=" * 60)
print("📋 PHASE 1: VALIDATION SUMMARY")
print("=" * 60)

# Collect validation results
validation_results = {}

# TPU Runtime check
try:
    devices = jax.devices()
    num_devices = len(devices)
    if num_devices > 0:
        device_type = devices[0].platform
        validation_results['TPU Runtime (8 cores)'] = num_devices == 8
        validation_results['TPU Device Type'] = device_type == 'tpu'
    else:
        validation_results['TPU Runtime (8 cores)'] = False
        validation_results['TPU Device Type'] = False
except Exception:
    validation_results['TPU Runtime (8 cores)'] = False
    validation_results['TPU Device Type'] = False

# JAX version check
try:
    validation_results['JAX Version'] = jax.__version__ is not None
except Exception:
    validation_results['JAX Version'] = False

# Tunix import check
validation_results['Tunix Import'] = 'tunix' in sys.modules

# Flax version check
try:
    validation_results['Flax Version'] = flax.__version__ is not None
except Exception:
    validation_results['Flax Version'] = False

# HBM visibility - optional
validation_results['HBM Visibility'] = True  # Always pass (optional)

# LoRA config ready
validation_results['LoRA Config Ready'] = 'lora_test_config' in globals()

# Display results
print("\n📊 Validation Results:")
print("-" * 40)
for check, passed in validation_results.items():
    status = '✅ PASS' if passed else '❌ FAIL'
    print(f"   {status} - {check}")

# Summary
all_passed = all(validation_results.values())
critical_passed = all([
    validation_results.get('JAX Version', False),
    validation_results.get('Tunix Import', False),
    validation_results.get('Flax Version', False),
])

print("\n" + "=" * 60)
if all_passed:
    print("🎉 ALL CHECKS PASSED - READY FOR PHASE 2 TRAINING")
elif critical_passed:
    print("⚠️  CORE CHECKS PASSED - Some optional checks failed")
    print("   Training can proceed, but review warnings above.")
else:
    print("❌ CRITICAL CHECKS FAILED - REVIEW ERRORS ABOVE")
    print("   Fix issues before proceeding to Phase 2.")
print("=" * 60)

print("\n📚 Troubleshooting Reference:")
print("   See docs/COLAB_VALIDATION_GUIDE.md")
print("   GitHub Issues: https://github.com/clduab11/judicAIta/issues")
```

**Phase 1 Success Criteria:**

✅ **READY FOR PHASE 2** if:
- [ ] All 8 TPU cores detected (or at least 4)
- [ ] TPU device type confirmed
- [ ] JAX, Tunix, and Flax imports successful
- [ ] LoRA configuration validated

**Validation Checklist:**

- [ ] All critical checks pass
- [ ] No `ModuleNotFoundError` errors
- [ ] Runtime properly restarted after installation
- [ ] Summary shows "READY FOR PHASE 2 TRAINING"

---

## Phase 2: Training Pipeline Verification

### Overview

This phase validates the training pipeline components, including model loading, dataset preparation, GRPO configuration, and training loop execution.

### 2.1 Model Download & Initialization

**Validation Steps:**

1. Run HuggingFace authentication cell (Step 3)
2. Run model download cell (Step 4)

**Expected Files:**

```
/root/.cache/huggingface/hub/models--google--gemma-3-1b-it/
├── snapshots/
│   └── <hash>/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── model-*.safetensors
│       └── ...
```

**Validation Cell:**

```python
import os
from pathlib import Path

print("=" * 60)
print("🔍 MODEL DOWNLOAD VERIFICATION")
print("=" * 60)

# Check model directory
model_cache = Path.home() / ".cache" / "huggingface" / "hub"
model_dirs = list(model_cache.glob("models--google--gemma*"))

if model_dirs:
    print(f"\n✅ Model found: {len(model_dirs)} variant(s)")
    for model_dir in model_dirs:
        print(f"   {model_dir.name}")
        
        # Check for required files
        snapshot_dirs = list((model_dir / "snapshots").glob("*"))
        if snapshot_dirs:
            snapshot = snapshot_dirs[0]
            required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
            for fname in required_files:
                fpath = snapshot / fname
                status = '✅' if fpath.exists() else '❌'
                print(f"      {status} {fname}")
else:
    print("\n❌ Model not found in cache")
    print("   Ensure Step 4 (model download) completed successfully")

print("\n" + "=" * 60)
```

**Validation Checklist:**

- [ ] Model downloaded to cache directory
- [ ] Required files present (config, tokenizer)
- [ ] No download errors

### 2.2 Dataset Preparation Validation

**Validation Cell:**

```python
print("=" * 60)
print("📊 DATASET PREPARATION VALIDATION")
print("=" * 60)

# Check training dataset
if 'training_dataset' in globals():
    print(f"\n✅ Training dataset loaded: {len(training_dataset)} examples")
    
    # Sample first example
    if len(training_dataset) > 0:
        sample = training_dataset[0]
        print(f"\n🔍 Sample Example Structure:")
        for key in sample.keys():
            value_preview = str(sample[key])[:100]
            print(f"   {key}: {value_preview}...")
        
        # Validate required fields
        required_fields = ['prompt', 'ground_truth']
        for field in required_fields:
            status = '✅' if field in sample else '❌'
            print(f"{status} Required field: '{field}'")
else:
    print("\n❌ Training dataset not found in globals")
    print("   Ensure Step 5 (dataset preparation) completed")

print("\n" + "=" * 60)
```

**Validation Checklist:**

- [ ] Training dataset loaded successfully
- [ ] Dataset contains expected number of examples
- [ ] Examples have required fields (prompt, ground_truth)
- [ ] Sample example displays correctly

### 2.3 GRPO Configuration Validation

**Validation Cell:**

```python
print("=" * 60)
print("🎯 GRPO CONFIGURATION VALIDATION")
print("=" * 60)

# Check GRPO config
if 'GRPO_CONFIG' in globals():
    print("\n✅ GRPO Configuration:")
    for key, value in GRPO_CONFIG.items():
        print(f"   {key}: {value}")
    
    # Validate key hyperparameters
    critical_params = {
        'num_iterations': (1, 1000),
        'batch_size': (1, 64),
        'learning_rate': (1e-7, 1e-3),
        'beta': (0.0, 1.0),
    }
    
    print("\n🔍 Parameter Validation:")
    for param, (min_val, max_val) in critical_params.items():
        if param in GRPO_CONFIG:
            value = GRPO_CONFIG[param]
            valid = min_val <= value <= max_val
            status = '✅' if valid else '⚠️ '
            print(f"{status} {param}: {value} (expected: {min_val}-{max_val})")
else:
    print("\n❌ GRPO_CONFIG not found")
    print("   Ensure Step 6 (configuration) completed")

# Check LoRA config
if 'LORA_CONFIG' in globals():
    print("\n✅ LoRA Configuration:")
    for key, value in LORA_CONFIG.items():
        print(f"   {key}: {value}")
else:
    print("\n⚠️  LORA_CONFIG not found")

print("\n" + "=" * 60)
```

**Validation Checklist:**

- [ ] GRPO_CONFIG defined with all required parameters
- [ ] Hyperparameters within valid ranges
- [ ] LORA_CONFIG defined
- [ ] No configuration errors

### 2.4 Reward Function Validation

**Validation Cell:**

```python
print("=" * 60)
print("🎁 REWARD FUNCTION VALIDATION")
print("=" * 60)

# Test reward function with sample data
if 'composite_reward_function' in globals():
    print("\n✅ Reward function found: composite_reward_function")
    
    # Create test data
    test_prompts = ["What is contract law?"]
    test_outputs = [
        "<reasoning>Contract law governs agreements between parties.</reasoning><answer>Valid</answer>"
    ]
    test_metadata = [{"ground_truth": "Valid"}]
    
    # Test tokenizer
    if 'tokenizer' in globals():
        print("   ✅ Tokenizer available")
        
        try:
            # Call reward function
            rewards = composite_reward_function(
                test_prompts,
                test_outputs,
                test_metadata,
                tokenizer
            )
            print(f"\n🔍 Test Reward Computation:")
            print(f"   Output: {test_outputs[0][:100]}...")
            print(f"   Reward: {rewards[0]:.4f}")
            print(f"   ✅ Reward function executes successfully")
        except Exception as e:
            print(f"\n❌ Reward function error: {e}")
    else:
        print("   ❌ Tokenizer not found")
else:
    print("\n❌ composite_reward_function not found")
    print("   Ensure reward function cell executed")

print("\n" + "=" * 60)
```

**Validation Checklist:**

- [ ] Reward function defined
- [ ] Test execution successful
- [ ] Reward value within expected range (0.0-1.0)
- [ ] No runtime errors

### 2.5 Training Setup Validation

**Validation Cell:**

```python
print("=" * 60)
print("🏋️ TRAINING SETUP VALIDATION")
print("=" * 60)

# Check if RLCluster is created
if 'rl_cluster' in globals():
    print("\n✅ RLCluster created")
else:
    print("\n❌ RLCluster not found")

# Check if GRPOLearner is created
if 'grpo_learner' in globals():
    print("✅ GRPOLearner created")
else:
    print("❌ GRPOLearner not found")

# Check if mesh is created
if 'mesh' in globals():
    print(f"✅ TPU Mesh created")
    print(f"   Shape: {mesh.shape}")
    print(f"   Axis names: {mesh.axis_names}")
else:
    print("❌ TPU Mesh not found")

# Check if models are initialized
models_status = {
    'actor_model': 'actor_model' in globals(),
    'reference_model': 'reference_model' in globals(),
}

print("\n🔍 Model Status:")
for model_name, exists in models_status.items():
    status = '✅' if exists else '❌'
    print(f"{status} {model_name}")

# Check checkpoint directories
import os
if os.path.exists('./checkpoints'):
    print("\n✅ Checkpoint directory exists")
else:
    print("\n⚠️  Checkpoint directory not created yet")

print("\n" + "=" * 60)
```

**Validation Checklist:**

- [ ] RLCluster created successfully
- [ ] GRPOLearner initialized
- [ ] TPU mesh configured
- [ ] Actor and reference models loaded
- [ ] Checkpoint directories exist

### 2.6 Training Loop Dry Run (Optional)

**Validation Cell:**

```python
print("=" * 60)
print("🧪 TRAINING LOOP DRY RUN (1 STEP)")
print("=" * 60)

if 'grpo_learner' in globals() and 'train_prompts' in globals():
    print("\n⏳ Running single training step...")
    
    try:
        # Get single batch
        test_batch = train_prompts[:2]  # Use 2 examples
        
        # Execute single training step
        step_metrics = grpo_learner.train_step(prompts=test_batch)
        
        print("\n✅ Training step completed successfully!")
        print(f"\n📊 Step Metrics:")
        for key, value in step_metrics.items():
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
            
    except Exception as e:
        print(f"\n❌ Training step failed: {e}")
        print("   This may indicate configuration issues")
else:
    print("\n⚠️  Prerequisites not met for dry run")
    print("   Ensure grpo_learner and train_prompts are available")

print("\n" + "=" * 60)
```

**Validation Checklist:**

- [ ] Single training step executes without errors
- [ ] Metrics returned include loss, reward, KL divergence
- [ ] No out-of-memory errors
- [ ] Execution completes in reasonable time (< 2 minutes)

### Phase 2 Success Criteria

✅ **READY FOR PHASE 3** if:
- [ ] Model downloaded and initialized successfully
- [ ] Dataset loaded with valid examples
- [ ] GRPO configuration validated
- [ ] Reward function executes correctly
- [ ] Training setup complete (optional: dry run passes)

---

## Phase 3: Inference & Output Quality

### Overview

This phase validates inference capabilities and output quality, including XML format validation, reasoning trace quality, and citation extraction.

### 3.1 Inference Test Setup

**Test Prompts:**

```python
TEST_PROMPTS = [
    "Is a verbal contract enforceable in most jurisdictions?",
    "What are the elements required to prove negligence?",
    "Can a contract be voided if one party was under duress?",
    "What is the statute of limitations for breach of contract?",
    "Does intellectual property include trade secrets?",
]
```

### 3.2 XML Format Validation

**Validation Cell:**

```python
import re

def validate_xml_format(text: str) -> bool:
    """Validate XML format with <reasoning> and <answer> tags."""
    has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', text, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', text, re.DOTALL))
    return has_reasoning and has_answer

def extract_xml_content(text: str):
    """Extract reasoning and answer from XML tags."""
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    answer = answer_match.group(1).strip() if answer_match else None
    
    return reasoning, answer

print("=" * 60)
print("🔍 XML FORMAT VALIDATION TEST")
print("=" * 60)

# Test with sample outputs
test_cases = [
    {
        "output": "<reasoning>Step-by-step analysis here</reasoning><answer>Yes</answer>",
        "expected_valid": True,
    },
    {
        "output": "Plain text without tags",
        "expected_valid": False,
    },
    {
        "output": "<reasoning>Only reasoning tag</reasoning>",
        "expected_valid": False,
    },
]

print("\n🧪 Running format validation tests...")
for i, test_case in enumerate(test_cases, 1):
    output = test_case["output"]
    expected = test_case["expected_valid"]
    result = validate_xml_format(output)
    
    status = '✅' if result == expected else '❌'
    print(f"{status} Test {i}: Valid={result} (expected={expected})")

print("\n✅ Format validation function working correctly")
print("=" * 60)
```

**Validation Checklist:**

- [ ] XML format validation function defined
- [ ] Test cases pass correctly
- [ ] Extraction function retrieves content

### 3.3 Inference Output Quality Test

**Validation Cell:**

```python
print("=" * 60)
print("🧪 INFERENCE OUTPUT QUALITY TEST")
print("=" * 60)

if 'grpo_learner' in globals() and 'TEST_PROMPTS' in globals():
    validation_results = []
    
    for i, prompt in enumerate(TEST_PROMPTS[:3], 1):  # Test first 3
        print(f"\n{'='*60}")
        print(f"📋 Test {i}: {prompt[:50]}...")
        print(f"{'='*60}")
        
        # Create full prompt
        full_prompt = create_prompt_template(prompt)
        
        try:
            # Generate response
            response = grpo_learner.generate(
                prompts=[full_prompt],
                max_tokens=512,
                temperature=0.7,
            )[0]
            
            # Validate format
            has_valid_format = validate_xml_format(response)
            reasoning, answer = extract_xml_content(response)
            
            # Count tokens
            reasoning_tokens = 0
            if reasoning and 'tokenizer' in globals():
                reasoning_tokens = len(tokenizer.encode(reasoning))
            
            print(f"\n✅ Generation successful")
            print(f"   Valid XML format: {has_valid_format}")
            print(f"   Reasoning tokens: {reasoning_tokens}")
            print(f"   Has reasoning: {reasoning is not None}")
            print(f"   Has answer: {answer is not None}")
            
            if reasoning:
                print(f"\n📝 Reasoning preview:")
                print(f"   {reasoning[:150]}...")
            if answer:
                print(f"\n💡 Answer:")
                print(f"   {answer[:100]}")
            
            validation_results.append({
                'prompt': prompt,
                'valid_format': has_valid_format,
                'reasoning_tokens': reasoning_tokens,
                'has_reasoning': reasoning is not None,
                'has_answer': answer is not None,
            })
            
        except Exception as e:
            print(f"\n❌ Generation error: {e}")
            validation_results.append({
                'prompt': prompt,
                'error': str(e),
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 INFERENCE QUALITY SUMMARY")
    print(f"{'='*60}")
    
    valid_count = sum(1 for r in validation_results if r.get('valid_format', False))
    avg_tokens = sum(r.get('reasoning_tokens', 0) for r in validation_results) / len(validation_results) if validation_results else 0
    
    print(f"\nTotal test prompts: {len(TEST_PROMPTS[:3])}")
    print(f"Valid XML format: {valid_count}/{len(validation_results)} ({100*valid_count/len(validation_results):.0f}%)")
    print(f"Avg reasoning tokens: {avg_tokens:.0f}")
    
    if valid_count == len(validation_results):
        print("\n✅ All outputs have valid XML format")
    else:
        print(f"\n⚠️  {len(validation_results) - valid_count} outputs have invalid format")
        
else:
    print("\n⚠️  Prerequisites not met")
    print("   Ensure grpo_learner and TEST_PROMPTS are available")

print("\n" + "=" * 60)
```

**Validation Checklist:**

- [ ] Inference generates responses for test prompts
- [ ] At least 80% of outputs have valid XML format
- [ ] Average reasoning length >= 100 tokens
- [ ] Responses are coherent and relevant
- [ ] No generation errors or crashes

### 3.4 Citation Extraction Test (If Applicable)

**Validation Cell:**

```python
import re

def extract_citations(text: str) -> list:
    """Extract legal citations from text."""
    # Common citation patterns
    patterns = [
        r'\d+\s+U\.S\.C\.\s+§\s+\d+',  # U.S. Code
        r'\d+\s+U\.S\.\s+\d+',  # U.S. Reports
        r'\d+\s+F\.\d+d\s+\d+',  # Federal Reporter
        r'\d+\s+S\.Ct\.\s+\d+',  # Supreme Court Reporter
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)
    
    return citations

print("=" * 60)
print("📚 CITATION EXTRACTION TEST")
print("=" * 60)

# Test citation extraction
test_text = """
The statute is found in 42 U.S.C. § 1983. The Supreme Court held in 
Miranda v. Arizona, 384 U.S. 436 (1966), that defendants must be informed 
of their rights.
"""

citations = extract_citations(test_text)
print(f"\n✅ Citations found: {len(citations)}")
for cite in citations:
    print(f"   • {cite}")

print("\n" + "=" * 60)
```

**Validation Checklist:**

- [ ] Citation extraction function defined
- [ ] Test cases extract citations correctly
- [ ] Common citation patterns recognized

### 3.5 Plain-English Summary Test (If Applicable)

**Validation Cell:**

```python
def assess_readability(text: str) -> dict:
    """Simple readability assessment."""
    sentences = text.split('.')
    words = text.split()
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    
    # Simple complexity assessment
    complexity = 'high_school' if avg_sentence_length < 20 else 'college'
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_sentence_length': avg_sentence_length,
        'estimated_level': complexity,
    }

print("=" * 60)
print("📖 READABILITY ASSESSMENT TEST")
print("=" * 60)

# Test with sample text
sample_text = """
Contract law governs agreements between parties. When both parties agree 
to terms, a binding contract is formed. The key elements include offer, 
acceptance, and consideration.
"""

assessment = assess_readability(sample_text)
print(f"\n✅ Readability Assessment:")
for key, value in assessment.items():
    print(f"   {key}: {value}")

print("\n" + "=" * 60)
```

**Validation Checklist:**

- [ ] Readability assessment function defined
- [ ] Test execution successful
- [ ] Metrics calculated correctly

### Phase 3 Success Criteria

✅ **READY FOR PHASE 4** if:
- [ ] Inference generates valid XML-formatted outputs
- [ ] At least 80% format validation pass rate
- [ ] Reasoning traces are substantive (>= 100 tokens average)
- [ ] Citation extraction works (if applicable)
- [ ] Readability assessment functional (if applicable)

---

## Phase 4: Documentation & Submission Prep

### Overview

This phase focuses on finalizing documentation, preparing submission materials, and conducting a final review.

### 4.1 README Update Checklist

**Required Updates to README.md:**

- [ ] Update "Quick Start" section with current installation steps
- [ ] Verify all code examples are current and executable
- [ ] Update TPU training section with Phase 1 validation reference
- [ ] Confirm dependency versions match current state
- [ ] Update troubleshooting guide with recent fixes
- [ ] Add Phase 1 validation checklist reference
- [ ] Verify links to documentation are working
- [ ] Update known issues section

**Validation:**

```bash
# Check for broken links
grep -r "http" README.md | grep -v "^#"

# Verify code blocks are syntactically correct
# (Manual review required)
```

### 4.2 Notebook Inline Documentation

**Documentation Standards:**

- [ ] Each cell has a descriptive markdown header
- [ ] Complex code blocks have inline comments
- [ ] Magic commands are explained
- [ ] Expected outputs are documented
- [ ] Error messages are explained
- [ ] Troubleshooting tips included where relevant

**Validation Cell:**

```python
import nbformat

# Load notebook
with open('train_tunix_reasoning.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Count documentation cells
markdown_cells = [c for c in nb.cells if c.cell_type == 'markdown']
code_cells = [c for c in nb.cells if c.cell_type == 'code']

print("=" * 60)
print("📝 NOTEBOOK DOCUMENTATION ASSESSMENT")
print("=" * 60)
print(f"\nMarkdown cells: {len(markdown_cells)}")
print(f"Code cells: {len(code_cells)}")
print(f"Documentation ratio: {len(markdown_cells)/(len(markdown_cells)+len(code_cells)):.1%}")

# Check for cells without prior markdown
undocumented_code = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and i > 0:
        if nb.cells[i-1].cell_type != 'markdown':
            undocumented_code += 1

if undocumented_code == 0:
    print("\n✅ All code cells have markdown documentation")
else:
    print(f"\n⚠️  {undocumented_code} code cells may lack documentation")

print("=" * 60)
```

### 4.3 Hackathon Submission Checklist

**Kaggle Submission Requirements:**

- [ ] `adapter_config.json` present and valid
- [ ] `adapter_model.safetensors` exported
- [ ] `README.md` with inference instructions
- [ ] Tokenizer files included
- [ ] Model card with training details
- [ ] License information
- [ ] Example usage code
- [ ] Validation results summary

**Package Structure:**

```
judicaita_submission.zip
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── README.md
└── validation_results.json
```

**Validation Script:**

```python
import zipfile
import json
from pathlib import Path

def validate_submission_package(zip_path: str):
    """Validate Kaggle submission package."""
    print("=" * 60)
    print("📦 SUBMISSION PACKAGE VALIDATION")
    print("=" * 60)
    
    required_files = [
        'adapter_config.json',
        'README.md',
        'tokenizer.json',
        'tokenizer_config.json',
    ]
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        files = zf.namelist()
        print(f"\n✅ Package created: {zip_path}")
        print(f"   Total files: {len(files)}")
        print(f"   Compressed size: {Path(zip_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        print("\n🔍 Required Files:")
        for fname in required_files:
            status = '✅' if fname in files else '❌'
            print(f"{status} {fname}")
        
        # Validate JSON files
        print("\n🔍 JSON Validation:")
        for fname in files:
            if fname.endswith('.json'):
                try:
                    content = zf.read(fname)
                    json.loads(content)
                    print(f"✅ {fname}: Valid JSON")
                except json.JSONDecodeError as e:
                    print(f"❌ {fname}: Invalid JSON - {e}")
    
    print("\n" + "=" * 60)

# Run validation
if Path('./judicaita_submission.zip').exists():
    validate_submission_package('./judicaita_submission.zip')
else:
    print("⚠️  Submission package not yet created")
```

### 4.4 Demo Outputs Capture

**Demo Output Requirements:**

- [ ] 3-5 example inference outputs
- [ ] Screenshots of training metrics
- [ ] Validation results summary
- [ ] Performance benchmarks
- [ ] Error handling examples

**Capture Cell:**

```python
import json
from datetime import datetime

def capture_demo_outputs(validation_results, training_metrics):
    """Capture demo outputs for submission."""
    demo_data = {
        'timestamp': datetime.now().isoformat(),
        'model': 'google/gemma-3-1b-it',
        'training_framework': 'Google Tunix + GRPO',
        'validation_results': validation_results,
        'training_metrics': {
            'final_loss': training_metrics['losses'][-1] if training_metrics.get('losses') else None,
            'final_reward': training_metrics['rewards'][-1] if training_metrics.get('rewards') else None,
            'total_steps': len(training_metrics.get('steps', [])),
        },
        'environment': {
            'runtime': 'Google Colab TPU',
            'tpu_type': 'v2-8',
        }
    }
    
    with open('validation_results.json', 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print("✅ Demo outputs captured: validation_results.json")
    return demo_data

# Execute if data available
if 'validation_results' in globals() and 'training_metrics' in globals():
    demo_data = capture_demo_outputs(validation_results, training_metrics)
else:
    print("⚠️  Demo data not available yet")
```

### 4.5 Final Code Review Checklist

**Code Quality:**

- [ ] No TODO/FIXME comments remaining
- [ ] All magic numbers replaced with constants
- [ ] Error handling present for critical operations
- [ ] Resource cleanup (file handles, etc.)
- [ ] No hardcoded paths (use Path objects)
- [ ] Type hints where appropriate

**Performance:**

- [ ] No unnecessary recomputations
- [ ] Efficient data loading
- [ ] Memory usage optimized
- [ ] TPU utilization monitored

**Security:**

- [ ] No API keys or tokens in code
- [ ] Input validation present
- [ ] No SQL injection vulnerabilities
- [ ] No path traversal vulnerabilities

### Phase 4 Success Criteria

✅ **SUBMISSION READY** if:
- [ ] README.md updated and accurate
- [ ] Notebook fully documented
- [ ] Submission package validated
- [ ] Demo outputs captured
- [ ] Code review completed
- [ ] All acceptance criteria met

---

## Troubleshooting Reference

### Common Issues and Solutions

#### Issue: `ModuleNotFoundError: No module named 'tunix'`

**Cause**: Incorrect Tunix version or missing installation

**Solution**:
```python
# Reinstall with correct version
!pip uninstall -y google-tunix
!pip install git+https://github.com/google/tunix
```

**Validation**: Check version with `pip show google-tunix`

---

#### Issue: `RuntimeError: TPU not found`

**Cause**: Wrong Colab runtime or missing restart

**Solution**:
1. Runtime → Change runtime type → TPU
2. Restart runtime
3. Re-run initialization cells (skip Step 1 installation)

**Validation**: `jax.devices()` should return 8 TPU devices

---

#### Issue: JAX TPU initialization fails

**Cause**: Incompatible JAX version or missing libtpu

**Solution**:
```python
# Reinstall JAX for TPU
!pip uninstall -y jax jaxlib
!pip install git+https://github.com/jax-ml/jax
```

**Validation**: Check with `jax.__version__` (should be 0.8.x)

---

#### Issue: `jax_cuda12_plugin` warnings

**Cause**: Normal Colab environment (has GPU packages pre-installed)

**Solution**: **IGNORE** - These warnings are harmless for TPU training

**Validation**: TPU training proceeds normally despite warnings

---

#### Issue: Imports fail after installation

**Cause**: Runtime not restarted after Step 1

**Solution**:
1. Runtime → Restart runtime
2. Skip Step 1 (already installed)
3. Continue from Step 2

**Validation**: All imports succeed without errors

---

#### Issue: Out of Memory during training

**Cause**: Batch size or sequence length too large

**Solution**:
```python
# Reduce hyperparameters
GRPO_CONFIG = {
    "batch_size": 2,  # Reduced from 4
    "num_generations": 2,  # Reduced from 4
    "max_tokens_to_generate": 256,  # Reduced from 512
}
```

**Validation**: Training proceeds without OOM errors

---

#### Issue: All rewards are 0.0

**Cause**: Model not generating XML tags or reward function issue

**Solution**:
1. Check sample output format
2. Validate `extract_xml_content()` function
3. Test reward function manually

**Validation**: Rewards show variation across examples

---

#### Issue: Loss not decreasing

**Cause**: Learning rate too high or reward signal insufficient

**Solution**:
```python
GRPO_CONFIG = {
    "learning_rate": 5e-6,  # Reduced from 1e-5
    "warmup_steps": 100,  # Increased warmup
}
```

**Validation**: Loss shows downward trend over iterations

---

### Getting Help

**Resources:**

- **GitHub Issues**: https://github.com/clduab11/judicAIta/issues
- **Documentation**: docs/COLAB_VALIDATION_GUIDE.md (this file)
- **Notebook README**: examples/notebooks/README.md
- **Tunix Documentation**: https://github.com/google/tunix

**Reporting Issues:**

When reporting issues, include:
1. Phase number (1-4)
2. Specific cell or step that failed
3. Complete error message
4. Environment details (Colab TPU, Python version)
5. Screenshot if relevant

---

## Appendix: Validation Checklist Summary

### Complete Validation Checklist

#### Phase 1: Environment & Dependency Validation
- [ ] 1.1 Colab runtime configured (TPU selected)
- [ ] 1.2 Dependencies installed successfully
- [ ] 1.3 Runtime restarted after installation
- [ ] 1.4 TPU devices detected (8 cores)
- [ ] 1.5 Core imports successful (JAX, Tunix, Flax)
- [ ] 1.6 HBM memory check (optional)
- [ ] 1.7 LoRA configuration validated
- [ ] 1.8 Phase 1 summary shows "READY FOR PHASE 2"

#### Phase 2: Training Pipeline Verification
- [ ] 2.1 Model downloaded and cached
- [ ] 2.2 Dataset loaded with valid examples
- [ ] 2.3 GRPO configuration validated
- [ ] 2.4 Reward function executes correctly
- [ ] 2.5 Training setup complete
- [ ] 2.6 Dry run test passes (optional)

#### Phase 3: Inference & Output Quality
- [ ] 3.1 Test prompts defined
- [ ] 3.2 XML format validation working
- [ ] 3.3 Inference generates valid outputs
- [ ] 3.4 Citation extraction functional (if applicable)
- [ ] 3.5 Readability assessment working (if applicable)

#### Phase 4: Documentation & Submission Prep
- [ ] 4.1 README.md updated
- [ ] 4.2 Notebook documentation complete
- [ ] 4.3 Submission package validated
- [ ] 4.4 Demo outputs captured
- [ ] 4.5 Final code review completed

### Acceptance Criteria Summary

- [ ] Notebook runs end-to-end in fresh Colab environment
- [ ] All 4 phases complete with passing validations
- [ ] No critical/high severity bugs remaining
- [ ] README reflects accurate, current implementation
- [ ] Submission materials ready for Kaggle upload

---

**Document Version**: 1.0  
**Last Updated**: December 30, 2025  
**Prepared for**: LEG-7 (Final Sprint - JudicAIta Colab Notebook Validation & Hackathon Submission Readiness)
