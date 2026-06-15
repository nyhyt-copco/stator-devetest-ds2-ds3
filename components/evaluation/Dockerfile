ARG IMAGE_REGISTRY_HOST=585768152950.dkr.ecr.eu-west-2.amazonaws.com
ARG PULL_IMAGE_REGISTRY_REPOSITORY=physicsx
ARG BASE_IMAGE_INFIX=
FROM ${IMAGE_REGISTRY_HOST}/${PULL_IMAGE_REGISTRY_REPOSITORY}/python:3.12${BASE_IMAGE_INFIX}-base

ARG COMPONENT_DIRECTORY
WORKDIR /app/${COMPONENT_DIRECTORY}

# System dependencies needed for visualization libraries (i.e. Open3D in PXS)
# TODO: Remove when physicsx/python base image gets updated
USER root
RUN apk add --no-cache ffmpeg libsm libxext libx11 eudev-libs mesa-gl libgomp
USER 1000

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

USER root

