# build in host machine

# docker run in host

docker run --gpus all -it \
    -v /home/wilsxue.adm/AILLM/wilsonLLM/TensorRT-Edge-LLM:/workspace \
    -w /workspace \
    --name wilson_pytorch_alpa nvcr.io/nvidia/pytorch:25.12-py3 \
    bash    

# build application in docker
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr/include/x86_64-linux-gnu \
    -DCUDA_CTK_VERSION=13.1 \
    -DCUTE_DSL_ARTIFACT_TAG=sm_100 \
    -DENABLE_CUTE_DSL=ALL