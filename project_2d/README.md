Welcome to **npecage**, a Python package developed to support automated plant root analysis from images. This package streamlines root segmentation, measurement, and feature extraction for scientific experiments.

- Documentation: https://curly-adventure-propror.pages.github.io/

# Deployment API - Docker Setup Guide

This guide will walk you through how to run the Dockerized API on **any computer**, without needing to install Python, dependencies, or configure anything manually.

---

## Step-by-Step: Run the App on Another Computer

### 1. Install Docker

- Download and install Docker Desktop (for macOS) or Docker Engine (for Linux):  [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) For windows: [https://docs.docker.com/desktop/setup/install/windows-install/](https://docs.docker.com/desktop/setup/install/windows-install/)

- Verify the installation:

```bash
docker --version
```

### 2. Pull the Docker Image

- The image is available on Docker Hub:

```bash
docker pull molnmark04/deployment.api.main:latest
```

### 3. Run the Container

- Run the image with port mapping:

```bash
docker run -it -p 8000:8000 molnmark04/deployment.api.main:latest
```

### 4. Test the API

- Once running, open a browser and navigate to:

```bash
http://localhost:8000/docs
```

- You'll see the interactive Swagger UI to test endpoinst easily.


# Deployment CLI - Docker Setup Guide

This guide will walk you through how to run the Dockerized CLI on **any computer**, without needing to install Python, dependencies, or configure anything manually.

---

## Step-by-Step: Run the App on Another Computer

### 1. Install Docker

- Download and install Docker Desktop (for macOS) or Docker Engine (for Linux):  [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) For windows: [https://docs.docker.com/desktop/setup/install/windows-install/](https://docs.docker.com/desktop/setup/install/windows-install/)


- Verify the installation:

```bash
docker --version
```

### 2. Pull the Docker Image

- The image is available on Docker Hub:

```bash
docker pull thijnbakker/deployment.cli.main:1.0.2
```

### 3. Log into your account

- This will ensure that the next step will work:

```bash
docker login
```

### 4. Test the CLI

- Once logged in run the following to get all the values:

```bash
docker run thijnbakker/deployment.cli.main:1.0.2 --image_path test_image.png --model_path models/model-best-2.h5
```

- This will give all the endpoints


## Contributing

Contributions are welcome! Please contact our team to improve `npecage`.

*Developed by Group CV4 as part of the ADSAI course at Breda University of Applied Sciences.*
