# Use the Microsoft Dev Containers base image as a starting point
# FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev \
    mesa-utils x11-apps \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    tree \
    && rm -rf /var/lib/apt/lists/*
