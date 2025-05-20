## MY SETUP 

- Windows 11 
- Asus ZenBook Duo, Intel Core Ultra 9 285H with 32GB RAM 
- *GPU:* Intel® Arc™ 140T GPU; Peak TOPS: 77
- *NPU:* Intel AI Boost: TOPS: 13


### Software Prerequisites: 

- C++ Libraries
- Microsoft Visual Studio 2022 Community Edition (choose Desktop Development with C++) 
- CMake 4.0 (download 64bit msi file) 
- Latest Intel Drivers (for GPU support)


### Installating Libraries for Using Intel GPU/NPU: 

_Installing PyTorch 2.7 for the Intel GPU_:

`uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu`

_Add PyTorch 2.7 to the pyproject.toml file:_

`uv add torch torchvision torchaudio`

_Installing the Intel Extension for PyTorch:_

Note that I cannot get `uv` to install/add this extension, so here I am just doing a traditional pip install until I can figure a way to use `uv`.

`pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/`


### Further Notes:

When running the `intel_ipex_check`, the following warning occurs:

_Overriding a previously registered kernel for the same operator and the same dispatch key_

I need to look into this! 

### Sources: 

- [Intel Extension for PyTorch Installation Guide](https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.7.10%2Bxpu&os=windows&package=pip)

- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/xpu/2.7.10+xpu/index.html)

- [Intel Extension for PyTorch Tutorial with Code Examples (Intel)](https://intel.github.io/intel-extension-for-pytorch/xpu/2.7.10+xpu/tutorials/examples.html)

- [Intel Extension for PyTorch Tutorial with Code Examples (PyTorch)](https://docs.pytorch.org/tutorials/recipes/intel_extension_for_pytorch.html)

