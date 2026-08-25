FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0+PTX"
ENV CMAKE_CUDA_ARCHITECTURES="86;89;90;120"
ENV FORCE_CUDA=1
ENV MAX_JOBS=4
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV BEZIER_IGNORE_VERSION_CHECK=1
ENV BEZIER_NO_EXTENSION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    git build-essential cmake ninja-build pkg-config \
    libgmp3-dev libmpfr-dev libboost-all-dev libglm-dev libtbb-dev \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    libglvnd0 libglx0 libegl1 libgles2 \
    libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.10 -m venv /app/.venv

RUN pip install --upgrade pip "setuptools<81" wheel packaging pybind11 ninja

RUN pip install --no-cache-dir torch==2.7.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

RUN echo "$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib" > /etc/ld.so.conf.d/torch.conf && ldconfig

ARG CACHEBUST=1
RUN git clone -b dev https://github.com/mthodoris/dmesh2.git

WORKDIR /app/dmesh2

RUN git submodule update --init --recursive --remote

RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

RUN pip install --no-cache-dir --no-build-isolation \
    torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

RUN MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git"

# --- Build CGAL (provides libcgal_diffdt.a dependency for the CUDA extension) ---
RUN cd external/cgal \
    && mkdir -p build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && cmake --build . -j"${MAX_JOBS}"

# --- Build nvdiffrast ---
RUN cd external/nvdiffrast && pip install --no-cache-dir --no-build-isolation -e .

# --- Build DMesh2Renderer (needs GLM, installed via apt above) ---
RUN cd external/dmesh2_renderer && pip install --no-cache-dir --no-build-isolation -e .

# --- Build CGAL-dependent wrapper code -> cgal_wrapper/libcgal_diffdt.a ---
RUN cd cgal_wrapper \
    && cmake -DCMAKE_BUILD_TYPE=Release . \
    && cmake --build . -j"${MAX_JOBS}"

# --- Build DMesh++ CUDA extension (mindiffdt._C) ---
RUN pip install --no-cache-dir --no-build-isolation -e .

RUN python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
RUN python -c "import mindiffdt._C; print('mindiffdt._C ok')"
RUN python -c "import nvdiffrast.torch; print('nvdiffrast ok')"

ENTRYPOINT ["bash", "-c", "git pull origin dev && exec \"$@\"", "--"]
WORKDIR /app/dmesh2

CMD ["/bin/bash"]
