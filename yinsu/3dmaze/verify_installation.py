#!/usr/bin/env python3
"""
Verification script for 3dmaze environment setup
Run this after creating the conda environment and installing packages
"""

import sys


def check_package(package_name, import_name=None):
    """Check if a package can be imported"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name:20s} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} - MISSING ({str(e)})")
        return False


def check_torch_cuda():
    """Check PyTorch CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ PyTorch CUDA        - OK (Version: {torch.version.cuda})")
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"  PyTorch: {torch.__version__}")
        else:
            print(f"✗ PyTorch CUDA        - NOT AVAILABLE")
            print(f"  PyTorch: {torch.__version__} (CPU only)")
        return cuda_available
    except ImportError:
        print(f"✗ PyTorch CUDA        - PyTorch not installed")
        return False


def main():
    print("=" * 60)
    print("3D Maze Environment Verification")
    print("=" * 60)
    print()

    # Core packages
    print("Core Packages:")
    packages = [
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("pyyaml", "yaml"),
        ("opencv", "cv2"),
        ("tensorboard", "tensorboard"),
    ]

    all_ok = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_ok = False

    print()
    print("Deep Learning:")
    if not check_package("torch", "torch"):
        all_ok = False
    if not check_package("torchvision", "torchvision"):
        all_ok = False
    if not check_package("torchaudio", "torchaudio"):
        all_ok = False

    print()
    print("GPU Support:")
    if not check_torch_cuda():
        all_ok = False

    print()
    print("Reinforcement Learning:")
    rl_packages = [
        ("gym", "gym"),
        ("gymnasium", "gymnasium"),
        ("stable-baselines3", "stable_baselines3"),
        ("memory-maze", "memory_maze"),
    ]

    for pkg_name, import_name in rl_packages:
        if not check_package(pkg_name, import_name):
            all_ok = False

    print()
    print("Visualization & Utils:")
    util_packages = [
        ("pygame", "pygame"),
        ("pillow", "PIL"),
        ("imageio", "imageio"),
        ("notebook", "notebook"),
    ]

    for pkg_name, import_name in util_packages:
        if not check_package(pkg_name, import_name):
            all_ok = False

    print()
    print("=" * 60)
    if all_ok:
        print("✓ All packages installed successfully!")
        print("✓ Environment is ready for training!")
    else:
        print("✗ Some packages are missing. Please install them:")
        print()
        print("  Missing PyTorch? Run:")
        print("    pip3 install torch torchvision torchaudio")
        print()
        print("  Missing other packages? Run:")
        print("    pip install gymnasium gym memory-maze stable-baselines3 \\")
        print("                pygame pillow imageio imageio-ffmpeg notebook")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())