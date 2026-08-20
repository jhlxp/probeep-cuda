# syntax=docker/dockerfile:1.7

# This is the Docker source of the H20 environment validated by Test01.
# Load the matching local OCI archive before building when the base tag is
# unavailable from a registry:
#
#   docker load -i /data/nfs/xuzc/probeep-image/flagtree-nvidia-3.6-v2.tar
#   docker build \
#     -f tests/Test_01_RawData1_Eval20/docker/Dockerfile \
#     -t probeep-xzc:h20 .

ARG BASE_IMAGE=flagtree-nvidia:3.6-v2
FROM ${BASE_IMAGE}

ARG NVSHMEM_VERSION=3.3.9
ARG VCS_REF=probeep-xzc

LABEL org.opencontainers.image.title="ProbeEP Test01 H20 runtime" \
      org.opencontainers.image.description="CUDA 12.9, PyTorch 2.8, NVSHMEM 3.3.9 and in-tree SM90 extensions" \
      org.opencontainers.image.source="https://github.com/jhlxp/probeep-cuda" \
      org.opencontainers.image.revision="${VCS_REF}"

USER root

# Test01 job 218324 used the 3.3.9 wheel contents mounted at /opt/nvshmem.
# Install the same version in the image so runtime nodes require no host-side
# Python-package bind mount.
RUN PIP_CONSTRAINT= python3 -m pip install \
        --no-cache-dir \
        --no-deps \
        --force-reinstall \
        "nvidia-nvshmem-cu12==${NVSHMEM_VERSION}" \
    && rm -rf /opt/nvshmem \
    && ln -s "$(python3 -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations)))')" /opt/nvshmem

ENV NVSHMEM_DIR=/opt/nvshmem \
    LD_LIBRARY_PATH=/opt/nvshmem/lib:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=9.0 \
    CUDA_MODULE_LOADING=EAGER \
    MAX_JOBS=16 \
    PROBEEP_ROOT=/workspace/probeep-cuda \
    PROBEEP_TEST_DIR=/workspace/probeep-cuda/tests/Test_01_RawData1_Eval20

WORKDIR /workspace/probeep-cuda
COPY --chown=ubuntu:ubuntu . .

USER ubuntu

# Keep each backend in its own source tree. Test01 launches isolated worker
# processes and selects the matching tree through PYTHONPATH at runtime.
RUN cd src/deepep \
    && python3 setup.py build_ext --inplace \
    && rm -rf build

RUN cd src/deepep-moonep \
    && python3 setup.py build_ext --inplace \
    && rm -rf build

RUN cd src/deepep-probeep \
    && python3 setup.py build_ext --inplace \
    && rm -rf build

RUN PYTHONPATH=/workspace/probeep-cuda/src/deepep-probeep \
    python3 -c 'import deep_ep, deep_ep_cpp, torch; assert torch.version.cuda == "12.9"; assert torch.compiled_with_cxx11_abi(); assert deep_ep_cpp.is_sm90_compiled()'

CMD ["/bin/bash"]
