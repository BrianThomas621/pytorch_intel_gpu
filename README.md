## SETUP / INSTALLATION STEPS

*Prerequisite*: 

Ensure you have installed the latest Intel GPU drivers!!

#### Steps Taken: 

After the folder & virtual directory have been created: 

_Installing PyTorch 2.7 for the XPU_:

- `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu`

_Add PyTorch 2.7 to the pyproject.toml file:_

- `uv add torch torchvision torchaudio`

_Installing the Intel Extension for PyTorch:_

Note that I cannot use `uv` for this, so just do a traditional pip install until further notice:

- `pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/`

_Add OpenVINO:_
- `uv add openvino==2025.1.0`

## Further Notes:

When running the intel_ipex_check, the following warning occurs:

_Overriding a previously registered kernel for the same operator and the same dispatch key_

See: 
https://github.com/intel/intel-extension-for-pytorch/issues/309


## Sources: 

*Intel Extension for PyTorch Installation Guide*
https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.7.10%2Bxpu&os=windows&package=pip

*Intel Extension for PyTorch*
https://intel.github.io/intel-extension-for-pytorch/xpu/2.7.10+xpu/index.html

*Intel Extension for PyTorch Tutorial with Code Examples:*
https://intel.github.io/intel-extension-for-pytorch/xpu/2.7.10+xpu/tutorials/examples.html

https://docs.pytorch.org/tutorials/recipes/intel_extension_for_pytorch.html


*OpenVINO 2025 Installation Guide*
https://docs.openvino.ai/2025/get-started/install-openvino.html
