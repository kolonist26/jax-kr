# JAX 설치하기

JAX는 순수 파이썬으로 작성되어 있지만, XLA에 의존하며, 이는 `jaxlib` 패키지로 설치해야 합니다.
`pip` 또는 `conda`를 사용해 이진 패키지를 설치, [Docker 컨테이너](#docker-컨테이너-nvidia-gpu)를 사용,
또는 [소스로부터 JAX 빌드](developer.md#building-from-source)를 위해서는 다음 지침을 확인하세요.

## 지원되는 플랫폼

|            | Linux x86_64 | Linux aarch64           | Mac x86_64   | Mac ARM        | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|-------------------------|--------------|----------------|----------------|---------------------|
| CPU        | [예](#cpu)         | [예](#cpu) | [예](#cpu)          | [예](#cpu)            | [예](#cpu)            | [예](#cpu)                 |
| NVIDIA GPU | [예](#nvidia-gpu)               | [예](#nvidia-gpu) | 아니오           | 해당사항 없음            | 아니오             | [실험적](#nvidia-gpu)        |
| Google TPU | [예](#google-tpu)  | 해당사항 없음                     | 해당사항 없음          | 해당사항 없음            | 해당사항 없음            | 해당사항 없음                 |
| AMD GPU    | [실험적](#amd-gpu) | 아니오                      | 아니오           | 해당사항 없음                 | 아니오             | 아니오                  |
| Apple GPU  | 해당사항 없음                 | 아니오                      | [실험적](#apple-gpu) | [실험적](#apple-gpu)   | 해당사항 없음            | 해당사항 없음                 |

Linux(Ubuntu 20.04 이상) 및 macOS(10.12 이상) 플랫폼에서 `jaxlib` 설치 또는 빌드를 지원합니다.
또한 *실험적*으로 네이티브 Windows 지원도 있습니다.

Windows 사용자는 [Windows Subsystem for
Linux](https://docs.microsoft.com/en-us/windows/wsl/about)를 통해 CPU 및 GPU에서 JAX를 사용할 수 있으며,
대안으로 네이티브 Windows CPU만 지원을 사용할 수 있습니다.

## CPU

### pip 설치: CPU

현재 다음 운영 체제 및 아키텍처에 `jaxlib` 휠을 배포하고 있습니다:
* Linux, x86-64
* Mac, Intel
* Mac, ARM
* Windows, x86-64 (*experimental*)

랩탑에서 로컬 개발을 하는 데 유용할 수 있는 CPU 전용 버전의 JAX를 설치하려면, 다음을 실행할 수 있습니다

```bash
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

Windows에서는 이미 설치되어 있지 않다면
[Microsoft Visual Studio 2019 Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)을 설치해야 할 수도 있습니다.

다른 운영 체제 및 아키텍처는 소스로부터 빌드해야 합니다.
다른 운영 체제 및 아키텍처에서 pip로 설치를 시도하면 `jax`는 성공적으로 설치될 수 있지만 `jaxlib`가 `jax`와 함께 설치되지 않을 수 있으며, 이 경우 실행 시 실패할 수 있습니다.

## NVIDIA GPU

JAX는 SM 버전 5.2(Maxwell) 이상을 가진 NVIDIA GPU를 지원합니다.
Kepler 시리즈 GPU는 NVIDIA가 소프트웨어에서 Kepler GPU 지원을 중단함에 따라 더 이상 JAX에서 지원되지 않습니다.

먼저 NVIDIA 드라이버를 설치해야 합니다.
NVIDIA에서 제공하는 가장 최신 드라이버를 설치하는 것을 권장하지만, CUDA 12의 경우 드라이버 버전이 >= 525.60.13,
CUDA 11의 경우 Linux에서 >= 450.80.02 이어야 합니다. NVIDIA 드라이버를 쉽게 업데이트할 수 없는 클러스터 등에서 더 새로운 CUDA 툴킷을 이전 드라이버와 함께 사용해야 하는 경우, 이 목적으로 NVIDIA가 제공하는 [CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/)를 사용할 수 있습니다.

### pip 설치: GPU (CUDA, pip를 통한 설치, 더 쉬움)

NVIDIA GPU 지원과 함께 JAX를 설치하는 두 가지 방법이 있습니다: pip 휠에서 설치한 CUDA와 CUDNN을 사용하는 방법과,
자체 설치한 CUDA/CUDNN을 사용하는 방법입니다. pip 휠을 사용하여 CUDA와 CUDNN을 설치하는 것이 훨씬 쉽기 때문에,
이 방법을 강력히 권장합니다! 이 방법은 NVIDIA가 aarch64 CUDA pip 패키지를 출시하지 않았기 때문에 x86_64에서만 지원됩니다.

```bash
pip install --upgrade pip

# CUDA 12 설치
# 주의: 휠은 리눅스에서만 가능.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 설치
# 주의: 휠은 리눅스에서만 가능.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

JAX가 잘못된 버전의 CUDA 라이브러리를 감지할 경우, 확인해야 할 몇 가지 사항이 있습니다:
* `LD_LIBRARY_PATH`가 설정되지 않았는지 확인하세요. `LD_LIBRARY_PATH`는 CUDA 라이브러리를 오버라이드할 수 있습니다.
* JAX가 요청한 CUDA 라이브러리가 설치되어 있는지 확인하세요. 위의 설치 명령어를 다시 실행하면 작동해야 합니다.

### pip 설치: GPU (CUDA, 로컬에 설치, 더 어려움)

미리 설치된 CUDA를 사용하고 싶다면, 먼저 [CUDA](https://developer.nvidia.com/cuda-downloads)와 [CuDNN](https://developer.nvidia.com/CUDNN)을 설치해야 합니다.

JAX는 **Linux x86_64만을 위한** 미리 빌드된 CUDA 호환 휠을 제공합니다. 다른 운영 체제 및 아키텍처 조합도 가능하지만,
[소스로부터 빌드](developer.md#building-from-source)가 필요합니다.

사용하는 NVIDIA 드라이버 버전은 [CUDA toolkit's corresponding driver version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)보다 적어도 같거나 새로워야 합니다.
예를 들어, NVIDIA 드라이버를 쉽게 업데이트할 수 없는 클러스터에서 더 새로운 CUDA 툴킷을 이전 드라이버와 함께 사용해야 한다면,
이 목적으로 NVIDIA가 제공하는 [CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/)를 사용할 수 있습니다.

JAX는 현재 두 가지 CUDA 휠 변형을 제공합니다:
* CUDA 12.3, cuDNN 8.9, NCCL 2.16
* CUDA 11.8, cuDNN 8.6, NCCL 2.16

CUDA, cuDNN, 및 NCCL 설치의 주 버전이 일치하고, 부 버전이 동일하거나 더 새로운 경우 JAX 휠을 사용할 수 있습니다.
JAX는 라이브러리 버전을 확인하며, 충분히 새롭지 않은 경우 오류를 보고합니다.

NCCL은 선택적 종속성이며, 여러 GPU를 사용한 계산을 수행하는 경우에만 필요합니다.

설치하려면, 아래의 내용을 실행하세요.

```bash
pip install --upgrade pip

# CUDA 12와 cuDNN 8.9 이상과 호환되는 휠을 설치.
# 주의: 휠은 linux에서만 가능.
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11과 cuDNN 8.6 이상과 호환되는 휠을 설치.
# 주의: 휠은 linux에서만 가능.
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**이 `pip` 설치는 Windows에서 작동하지 않고 안내 없이 실패할 것입니다;
위의 [JAX 설치하기](#jax-설치하기)를 참고하세요.**

아래 명령어를 통해 CUDA 버전을 확인할 수 있습니다:

```bash
nvcc --version
```

JAX는 CUDA 라이브러리를 찾기 위해 `LD_LIBRARY_PATH`를 사용하고 바이너리(`ptxas`, `nvlink`)를 찾기 위해 `PATH`를 사용합니다.
이 경로들이 올바른 CUDA 설치를 가리키는지 확인해 주세요.

미리 빌드된 휠과 관련하여 오류나 문제가 발생하는 경우, [이슈 트래커](https://github.com/google/jax/issues)를 통해 알려주세요.

### Docker 컨테이너: NVIDIA GPU

NVIDIA는 [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox) 컨테이너를 제공하며,
이는 JAX의 나이틀리 릴리즈와 일부 모델/프레임워크를 포함한 최신 컨테이너입니다.

## 나이틀리 설치

나이틀리 릴리스는 빌드된 시점에서 메인 저장소의 상태를 반영하며, 전체 테스트 스위트를 통과하지 않을 수 있습니다.

* JAX:
```bash
pip install -U --pre jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
```

* Jaxlib CPU:
```bash
pip install -U --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
```

* Jaxlib TPU:
```bash
pip install -U --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
pip install -U --pre libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

* Jaxlib GPU (Cuda 12):
```bash
pip install -U --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda12_releases.html
```

* Jaxlib GPU (Cuda 11):
```bash
pip install -U --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda_releases.html
```

## Google TPU

### pip 설치: Google Cloud TPU

JAX는 [Google Cloud TPU](https://cloud.google.com/tpu/docs/users-guide-tpu-vm)를 위한 미리 빌드된 휠을 제공합니다.
적절한 버전의 `jaxlib` 및 `libtpu`와 함께 JAX를 설치하려면, 클라우드 TPU VM에서 다음을 실행할 수 있습니다:
```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

대화형 노트북 사용자를 위해: Colab TPU는 JAX 버전 0.4부터 JAX를 더 이상 지원하지 않습니다.
그러나 클라우드에서 대화형 TPU 노트북을 사용하고자 한다면, JAX를 완전히 지원하는 [Kaggle TPU notebooks](https://www.kaggle.com/docs/tpu)을 사용할 수 있습니다.

## Apple GPU

### pip 설치: Apple GPUs

Apple은 Apple GPU 하드웨어를 위한 실험적인 Metal 플러그인을 제공합니다.
자세한 내용은 [Apple의 JAX on Metal 문서](https://developer.apple.com/metal/jax/)를 참고하세요.

Metal 플러그인과 관련된 몇 가지 주의사항이 있습니다:
* Metal 플러그인은 새롭고 실험적이며
  [알려진 문제들](https://github.com/google/jax/issues?q=is%3Aissue+is%3Aopen+label%3A%22Apple+GPU+%28Metal%29+plugin%22)이 여러 개 있습니다.
  문제가 발생하면 JAX 이슈 트래커에 보고해 주세요.
* Metal 플러그인은 현재 매우 구체적인 버전의 `jax`와 `jaxlib`을 요구합니다.
  이 제한은 시간이 지나면서 플러그인 API가 성숙해짐에 따라 완화될 것입니다.

## AMD GPU

JAX has experimental ROCM support. There are two ways to install JAX:

* [AMD's docker 컨테이너](https://hub.docker.com/r/rocm/jax)를 사용하거나,
* [소스로부터 빌드](developer.md#additional-notes-for-building-a-rocm-jaxlib-for-amd-gpus)하세요.

## Conda

### Conda 설치

`jax`의 커뮤니티 지원 빌드가 있습니다. `conda`를 통해 설치하려면, 다음을 실행하세요.

```bash
conda install jax -c conda-forge
```

NVIDIA GPU가 있는 기계에 설치하려면, 다음을 실행하세요.
```bash
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

`conda-forge`에 의해 배포된 `cudatoolkit`은 JAX가 필요로 하는 `ptxas`가 누락되어 있습니다.
따라서 `nvidia` 채널에서 `cuda-nvcc` 패키지를 설치하거나, `ptxas`가 경로에 있도록 기계에 CUDA를 별도로 설치해야 합니다.
위의 채널 순서(`conda-forge` 이전에 `nvidia`)가 중요합니다.

JAX에 의해 사용되는 CUDA의 릴리스를 오버라이드하고 싶거나, GPU가 없는 기계에 CUDA 빌드를 설치하려면, `conda-forge` 웹사이트의 [Tips & tricks](https://conda-forge.org/docs/user/tipsandtricks.html#installing-cuda-enabled-packages-like-tensorflow-and-pytorch) 섹션의 지침을 따르세요.

자세한 내용은 `conda-forge`의 `jaxlib` 및 `jax` 저장소를 참고하세요.

## 소스로부터 JAX 빌드
[소스로부터 JAX 빌드](developer.md#building-from-source)를 참고하세요.

## 이전 jaxlib 휠 설치하기

Python 패키지 인덱스의 저장 공간 제한으로 인해, 우리는 주기적으로 http://pypi.org/project/jax에서 이전 jaxlib 휠을 제거합니다.
이들은 여기 있는 URL을 통해 직접 설치할 수 있습니다; 예를 들어:
```
# 휠 아카이브를 통해 CPU용 jaxlib 설치
pip install jax[cpu]==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# jaxlib 0.3.25 CPU 휠을 직접 설치
pip install jaxlib==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
특정 이전 GPU 휠을 사용하려면 `jax_cuda_releases.html` URL을 사용하세요; 예를 들어
```
pip install jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
