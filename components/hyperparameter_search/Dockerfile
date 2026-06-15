ARG IMAGE_REGISTRY_HOST=585768152950.dkr.ecr.eu-west-2.amazonaws.com
ARG PULL_IMAGE_REGISTRY_REPOSITORY=physicsx
ARG BASE_IMAGE_INFIX=
FROM ${IMAGE_REGISTRY_HOST}/${PULL_IMAGE_REGISTRY_REPOSITORY}/python:3.12${BASE_IMAGE_INFIX}-cuda-12.6

ARG COMPONENT_DIRECTORY
WORKDIR /app/${COMPONENT_DIRECTORY}

# Optional system dependencies for visualization libraries
# Uncomment if you encounter errors related to:
# - ffmpeg: Required for video/animation exports (matplotlib FFMpegWriter, PyVista movie exports)
# - libsm6/libxext6: Required if X11-linked matplotlib/VTK wheels fail to import with libSM.so.6 or libXext.so.6 errors
# Current headless usage does not seem to require these packages.
#
# USER root
# RUN apt-get update && \
#     apt-get install -y ffmpeg libsm6 libxext6 && \
#     rm -rf /var/lib/apt/lists/*
# USER 1000

# Make sure logs are printed immediately
ENV PYTHONUNBUFFERED=1

# Set location of artifactory credentials
ENV NETRC=/run/secrets/.netrc

# Disable Ray's UV runtime environment hook to prevent workers from re-syncing dependencies
# see https://github.com/ray-project/ray/releases/tag/ray-2.47.0#:~:text=uv%20run%20integration,0%20environment%20variable
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Copy the commons library (if it exists)
COPY libraries/ /app/libraries/

# Install python dependencies
RUN --mount=type=cache,target=.cache/uv \
    --mount=type=secret,id=.netrc,uid=1000 \
    --mount=type=bind,source=${COMPONENT_DIRECTORY}/pyproject.toml,target=/app/${COMPONENT_DIRECTORY}/pyproject.toml \
    --mount=type=bind,source=${COMPONENT_DIRECTORY}/uv.lock,target=/app/${COMPONENT_DIRECTORY}/uv.lock \
    uv sync --compile-bytecode --no-install-project --locked

# Copy only the specific component
COPY ${COMPONENT_DIRECTORY}/ /app/${COMPONENT_DIRECTORY}

# Sync the project
RUN --mount=type=cache,target=.cache/uv \
    --mount=type=secret,id=.netrc,uid=1000 \
    uv sync --locked --compile-bytecode

# Set nvidia devices
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Add if performing distributed training, otherwise performance may be degraded
# ENV OMP_NUM_THREADS=1

USER root

