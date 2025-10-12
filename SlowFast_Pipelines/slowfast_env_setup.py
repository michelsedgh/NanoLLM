import importlib.util
import os
import subprocess
import sys

try:
    from packaging import version as pkg_version
except Exception:  # pragma: no cover - packaging is part of base images
    pkg_version = None

try:
    import pkg_resources
except Exception:  # pragma: no cover - pkg_resources is part of setuptools
    pkg_resources = None

# Global list to store failed commands
failed_commands = []


# Packages we must never install/upgrade from this script
# (leave to the base container):
_BLOCKED_PIP_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
    "torchtext",
    "triton",
}

def _module_available(module_name):
    return importlib.util.find_spec(module_name) is not None


def _distribution_version(dist_name):
    if pkg_resources is None:
        return None
    try:
        return pkg_resources.get_distribution(dist_name).version
    except Exception:
        return None


def ensure_python_package(pip_name, module_name=None, min_version=None, install_cmd=None):
    """Verify package availability before installing to avoid overwriting Jetson stacks."""
    module_name = module_name or pip_name.replace("-", "_")
    installed = _module_available(module_name)

    if installed:
        if min_version and pkg_version:
            current = _distribution_version(pip_name)
            if current and pkg_version.parse(current) < pkg_version.parse(min_version):
                print(
                    f"Found {pip_name}=={current} (<{min_version}); attempting to upgrade conservatively."
                )
            else:
                print(f"✓ {pip_name} already satisfies the requirement.")
                return True
        else:
            print(f"✓ {pip_name} already available (module '{module_name}').")
            return True

    # Never install/upgrade core torch packages here. Keep container's versions.
    base_pip_name = pip_name.split("[")[0].split("==")[0]
    if base_pip_name in _BLOCKED_PIP_PACKAGES:
        print(f"⏭ Skipping install/upgrade for '{pip_name}' (managed by base image).")
        return True

    if install_cmd is None:
        install_cmd = f"{sys.executable} -m pip install {pip_name}"

    return run_command(install_cmd)

def run_command(command):
    """
    Runs a shell command and prints its output.
    Records failures but does not exit.
    """
    print(f"Executing: {command}")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
        process.wait() # Wait for the subprocess to complete
        if process.returncode != 0:
            print(f"Warning: Command '{command}' failed with exit code {process.returncode}")
            failed_commands.append(command)
            return False # Indicate failure
        return True # Indicate success
    except Exception as e:
        print(f"Warning: An exception occurred while trying to run command '{command}': {e}")
        failed_commands.append(command)
        return False # Indicate failure

def install_dependencies():
    """
    Installs all the required Python dependencies, continuing on failure.
    """
    print("--- Starting dependency installation ---")

    # Verify core runtime (Python, torch, torchvision) already provided by container.
    print("\n--- Verifying core runtime ---")
    ensure_python_package("torch", module_name="torch")
    ensure_python_package("torchvision", module_name="torchvision")
    ensure_python_package("torchaudio", module_name="torchaudio")

    # Core Python dependencies (installed only when missing)
    print("\n--- Ensuring core Python dependencies ---")
    for pkg in [
        ("numpy", "numpy"),
        ("simplejson", "simplejson"),
        ("psutil", "psutil"),
        ("tqdm", "tqdm"),
        ("PyYAML", "yaml"),
        ("iopath", "iopath"),
        ("tensorboard", "tensorboard"),
        ("moviepy", "moviepy"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("plotly", "plotly"),
    ]:
        ensure_python_package(pkg_name := pkg[0], module_name=pkg[1])

    # OpenCV is best provided via apt on Jetson; warn if missing.
    print("\n--- Checking OpenCV availability ---")
    if not _module_available("cv2"):
        print(
            "⚠ cv2 not found. Please install 'python3-opencv' via apt inside the container if needed."
        )
    else:
        print("✓ OpenCV (cv2) detected.")

    # Install fairscale / fvcore only if missing.
    print("\n--- Ensuring fairscale and fvcore ---")
    ensure_python_package(
        "git+https://github.com/facebookresearch/fairscale",
        module_name="fairscale",
        install_cmd=f"{sys.executable} -m pip install --no-deps git+https://github.com/facebookresearch/fairscale",
    )
    ensure_python_package(
        "git+https://github.com/facebookresearch/fvcore.git",
        module_name="fvcore",
        install_cmd=f"{sys.executable} -m pip install --no-deps git+https://github.com/facebookresearch/fvcore.git",
    )

    # Install cocoapi only if pycocotools missing.
    print("\n--- Ensuring COCO API (pycocotools) ---")
    ensure_python_package(
        "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI",
        module_name="pycocotools",
        install_cmd=(
            f"{sys.executable} -m pip install "
            "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
        ),
    )

    # Install detectron2 when requested and missing.
    print("\n--- Ensuring Detectron2 ---")
    ensure_python_package(
        "git+https://github.com/facebookresearch/detectron2@7c2c8fb",
        module_name="detectron2",
        install_cmd=(
            f"{sys.executable} -m pip install --no-deps "
            "git+https://github.com/facebookresearch/detectron2@7c2c8fb"
        ),
    )

    # Install pytorchvideo if missing.
    print("\n--- Ensuring PyTorchVideo ---")
    ensure_python_package(
        "git+https://github.com/facebookresearch/pytorchvideo.git",
        module_name="pytorchvideo",
        install_cmd=(
            f"{sys.executable} -m pip install --no-deps "
            "git+https://github.com/facebookresearch/pytorchvideo.git"
        ),
    )

    # the user will set it in their environment before running their main application if needed for slowfast)
    # For a persistent effect within the script's execution context, we can modify os.environ
    print("\n--- Setting PYTHONPATH for slowfast (if applicable) ---")
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    # Get the directory of the current script, then assume slowfast is a sibling directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slowfast_path = os.path.join(script_dir, 'slowfast')
    # Make sure to expanduser to handle '~' if it were ever used (though less common with os.getcwd())
    slowfast_path = os.path.abspath(slowfast_path)

    # Only add if it's not already in the PYTHONPATH
    if slowfast_path not in current_pythonpath.split(os.pathsep):
        os.environ['PYTHONPATH'] = f"{slowfast_path}{os.pathsep}{current_pythonpath}" if current_pythonpath else slowfast_path
        print(f"Updated PYTHONPATH: {os.environ['PYTHONPATH']}")
    else:
        print("slowfast path already in PYTHONPATH.")


    # Install specific ultralytics and Pillow
    print("\n--- Ensuring ultralytics and Pillow ---")
    ensure_python_package(
        "ultralytics",
        "ultralytics",
        install_cmd=f"{sys.executable} -m pip install --no-deps ultralytics",
    )
    ensure_python_package(
        "Pillow==9.5.0",
        module_name="PIL",
        min_version="9.5.0",
        install_cmd=f"{sys.executable} -m pip install Pillow==9.5.0",
    )

    print("\n--- Dependency installation attempt complete ---")

    # Summary of failures
    print("\n--- Installation Summary ---")
    if failed_commands:
        print("\nThe following commands failed during execution:")
        for cmd in failed_commands:
            print(f"- {cmd}")
        print("\nPlease review the output above for specific error messages for these commands.")
    else:
        print("\nAll commands completed successfully (though individual package installations might still have warnings).")

    # Verify CUDA availability
    print("\n--- Verifying PyTorch CUDA availability ---")
    try:
        import torch
        print(f"Torch can access CUDA : {torch.cuda.is_available()}")
    except ImportError:
        print("Error: torch could not be imported. CUDA availability check skipped.")
    except Exception as e:
        print(f"An unexpected error occurred during CUDA check: {e}")


if __name__ == "__main__":
    install_dependencies()
