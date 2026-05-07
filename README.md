# `stator-devtest-nc-ds2` Project

A project for building and running machine learning workflows.

Generated using:

- `flux_pipelines 0.60.0`
- `flux-templates 1.4.1`

## Structure

The project is split into 5 main sections:

- `components`, a directory defines individual components as packages which you can use in your pipelines.
- `pipelines`, which allows you to create custom pipelines by importing your prebuilt components.
- `services`, a directory for KService deployments that provide HTTP endpoints for model inference.
- `libraries`, a directory for any packages you wish to share across multiple components. We have included an example `commons` library.
- `conf`, which contains all the parameters to run each pipeline. These can be either JSON or YAML files.

## Getting started

### Initialize Git repository (recommended)

If you haven't already, initialize this project as a git repository to track your changes:

```bash
git init
git add .
git commit -m "Initial commit: stator-devtest-nc-ds2 project setup"
```

You can add a remote repository later when you're ready to share your code:

```bash
git remote add origin <your-repository-url>
git push -u origin main
```

### Task Overview

Task is our way of reducing project commands to one file. It's very similar to Makefiles, but designed for running installations, tests and other useful tasks rather than building C objects. Task comes pre-installed in the platform.

### Setup artifactory credentials and `uv`

> We recommend using `uv` as a package manager for this project, as developing pipelines in separate environments from components necessitates changing virtual environment regularly. `uv` facilitates this by running commands in the appropriate environment for a directory __when each command is prefixed with `uv run ...`__. Note, this is not required for any `task` commands. If you wish to use another package manager, such as `conda` or `pip`, we recommend creating a single environment and installing all dependencies in it from every component project.

#### Setup artifactory credentials

First, set up your artifactory credentials using the flux-pipelines CLI:

```bash
flux-pipelines auth setup
```

This command will prompt you for your artifactory username and access token, then create a `.netrc` file in your home directory with the proper credentials and permissions.

### Install the project

Installing the project can be done with:

```bash
$ task install

Using CPython 3.11.5 interpreter at: ...
Creating virtual environment at: .venv
Resolved 225 packages in 24ms
Installed 190 packages in 5.09s
```

Now you should be able to run, publish and list pipelines available to you.

## Running commands

### Pipelines

First thing to do is to list all the available pipelines. We've included an example pipeline here for you:

```bash
$ flux-pipelines pipeline list
                                   Local Pipelines
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name        ┃ Description                                                         ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ hello_world │ Prints 'Hello ${ config.name }!' in a custom component.             │
└─────────────┴─────────────────────────────────────────────────────────────────────┘
```

This pipeline comes with a prebuilt `hello_world` component, which you should be able to run straight away:

```bash
$ flux-pipelines pipeline run -p hello_world -e test --data-path conf/hello_world/example.yaml

Run details available from: /pipeline/#/runs/details/377f8be9-8cae-4fce-87c6-c8154239f232
```

Note a few things with this command:

- `-p` signifies which pipeline to run.
- `-e` is the experiment to run the pipeline (we will also create a default experiment for this)
- `--data-path` defines where the parameters of the pipeline are located. In this case, we've added an example configuration in `conf/hello_world/example.yaml`.

You can also publish the pipeline to the AI Workbench UI, so other users can run them with their own configuration in a low-code environment:

```bash
$ flux-pipelines pipeline publish -p hello_world

Pipeline details available from: 1472fe5f-4b13-47da-b886-c504c3749152
```

## Building new pipelines

To create a new pipeline, run:

```bash
$ flux-pipelines pipeline create

? What is the name of your new pipeline? ...
? Select which template to use. (Use arrow keys)
 » hello_world
   training
```

This will generate a new template for you and auto-register it with the CLI. The `hello_world` template provides a simple example, while the `training` template shows a more complex ML training workflow.

## Building new components

To create a new component, you can run:

```bash
$ flux-pipelines component create

? What is the name of your new component? test
? Select which template to use. (Use arrow keys)
 » evaluation
   importer
   modelling
   preprocessing
   splitting
```

This will generate a new template for you in the `components` directory. Each template type takes different inputs and outputs according to what is typically required for each step of a machine learning workflow. You can choose between CPU or GPU docker images based on your component's requirements.

Each template will come with a:

- `component.py`, where the core logic of the component is defined through PXC.
- `debug_component.py`, which can be used for checking the behaviour of the component is correct.
- `Dockerfile`, which defines how to package the code up into images to be deployed in the cluster.
- `pyproject.toml`, defining the name of the component, the version, dependencies and any specific configuration for development.

Don't forget to add the component to the CI build in your `project-components-build-push.yaml`:

```yaml
jobs:

  hello_world:
    name: hello_world
    uses: ./.github/workflows/components-build-push.yaml
    secrets: inherit

    permissions:
      actions: read
      id-token: write
      contents: write

    with:
      component-name: hello_world
      package-directory: examples/project
      component-directory: components/hello_world
      component-repository: components
      dockerfile: examples/project/components/hello_world/Dockerfile
```

Now, every time the version of your component is bumped, the CI will build a new docker image for you. You can import these prebuilt components from the `prebuilt` directory under each component and use them directly in your pipelines:

```python
from kfp import dsl
from typing import Dict

from components.my_new_component.prebuilt.v0_0_1 import component as my_new_component

@dsl.pipeline(name="my_pipeline")
def my_pipeline(config: Dict):
    task = my_new_component(config=config)
```

# Components Guide

## Creating a New Component

### 1. Generate Component Template

To create a new component, run:

```bash
$ flux-pipelines component create

? What is the name of your new component? my_new_component
? Select which template to use. (Use arrow keys)
» importer
  preprocessing
  splitting
  modelling
  evaluation
  hello_world
? Select which docker image to use. (Use arrow keys)
» cpu
  gpu
```

### 2. Component Templates

Components are the building blocks of your pipelines. The available templates (`importer`, `preprocessing`, `splitting`, `modelling`, `evaluation`, `hello_world`) provide different functionality for machine learning workflows. Each component follows a standard structure with `component.py`, `debug_component.py`, `Dockerfile`, and `pyproject.toml`.

For detailed examples of all component types and their usage, see the **Training Pipeline** section below which demonstrates how these components work together in a complete ML workflow.

### 3. Component Structure

Each generated component contains:

```
components/my_new_component/
├── component.py           # Main component logic (PXC-based)
├── debug_component.py     # Local testing script
├── Dockerfile            # Container definition
├── pyproject.toml        # Project metadata and dependencies
└── prebuilt/             # Auto-generated prebuilt versions (via CI)
```

### 4. Training Pipeline Template

The training pipeline template provides a complete ML workflow that uses all the component templates together. It includes:

- **Complete ML Pipeline**: Data import → preprocessing → splitting → model training → evaluation
- **Pre-configured Components**: All components are pre-built and ready to use
- **Resource Management**: Optimized CPU/GPU resource allocation for each step
- **MLflow Integration**: Automatic experiment tracking and model registry
- **Default Configuration**: Works with battery surface data out of the box

#### Components in Training Pipeline:

1. **Importer**: Loads battery surface and coolant data from PyVista datasets
2. **Preprocessing**: Surface merging, field transformations, and decimation
3. **Splitting**: Random 80/10/10 train/validation/test split
4. **Modelling**: OPORA neural network training with uncertainty quantification on GPU
5. **Evaluation**: Performance metrics and scatter plot visualization

#### Default Configuration:

The training pipeline comes with a complete configuration file (`conf/training/battery.yaml`) that includes:

```yaml
# Key default settings
modelling_config:
  device: cuda                    # GPU training
  model:
    model_type: opora
    architecture: pct             # Point Cloud Transformer
    width: 128                    # Hidden layer size
    dropout: 0.05
    training:
      n_epochs: 50
      batch_size: 1
      learning_rate: 0.001

preprocessing_config:
  layers:
    - type: merge_surfaces        # Combine coolant surfaces
    - type: convert_cell_to_point_data
    - type: transform_field       # Log pressure transformation
    - type: decimate_surface      # 95% surface reduction

splitting_config:
  split_type: random
  datasets:
    train: 0.8
    validation: 0.1
    test: 0.1
```

#### Running the Training Pipeline:

```bash
$ flux-pipelines pipeline run -p training -e battery-experiment --data-path conf/training/battery.yaml
```

This provides a complete example of how all component types work together in production.

## Running Pipelines

#### List Available Pipelines


```bash
$ flux-pipelines pipeline list
                                  Local Pipelines
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name        ┃ Description                                                         ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ hello_world │ Prints 'Hello ${ config.name }!' in a custom component.             │
│ training    │ Complete ML pipeline: data import → training → evaluation           │
└─────────────┴─────────────────────────────────────────────────────────────────────┘
```

#### Run Pipeline

```bash
$ flux-pipelines pipeline run -p hello_world -e test --data-path conf/hello_world/example.yaml

Run details available from: /pipeline/#/runs/details/377f8be9-8cae-4fce-87c6-c8154239f232
```

Command breakdown:
- `-p hello_world`: Specifies which pipeline to run
- `-e test`: Experiment name for organizing runs
- `--data-path conf/hello_world/example.yaml`: Configuration file with parameters

#### Publish Pipeline to AI Workbench UI

Make pipelines available to other users in low-code environment:

```bash
$ flux-pipelines pipeline publish -p hello_world

Pipeline details available from: 1472fe5f-4b13-47da-b886-c504c3749152
```

#### Monitor Pipeline Execution

- **AI Workbench UI**: Use the URL from run output to view execution in browser
- **CLI Status**: Check pipeline status and logs through AI Workbench interface
- **Artifacts**: Access output artifacts through the pipeline run interface

# Services Guide

Services provide HTTP endpoints for model inference using the KService framework.


## Creating a New Service

### 1. Generate Service Structure

You can create a new service using the CLI, which supports different templates for different use cases:

#### Using flux-pipelines CLI (Recommended)

**Interactive Mode:**
```bash
$ flux-pipelines service create

? What is the name of your new service? my_new_service
? Select which template to use: (Use arrow keys)
» basic
  mlflow
```

**Command Line Mode:**
```bash
# Basic service template
$ flux-pipelines service create --name my_service --template basic

# MLflow service template (for model serving)
$ flux-pipelines service create \
  --name my_model_service \
  --template mlflow \
  --model-name my-model \
  --model-version 1
```

#### Available Service Templates:

- **`basic`**: Simple KService template for general HTTP endpoints (uses lightweight CPU-only image for efficiency)
- **`mlflow`**: Specialized template for serving MLflow models with automatic model loading from the catalog (uses GPU-enabled image for optimal ML inference performance)

#### Manual creation (Not recommended)
```bash
mkdir -p services/my_new_service
cd services/my_new_service
```

## MLflow Model Service

The MLflow template creates a service that automatically loads models from the MLflow catalog using environment variables. This is perfect for serving trained models from your pipelines.

### MLflow Service Features

- **Automatic Model Loading**: Loads models from MLflow catalog using `models:/{model_name}/{model_version}` URI
- **Environment Configuration**: Model name and version specified via environment variables
- **Model Unwrapping**: Automatically handles PXS model unwrapping for proper inference
- **GPU Support**: Pre-configured for GPU inference
- **Type Safety**: Includes proper Pydantic schemas for request/response validation

### Example: Creating an MLflow Service

```bash
$ flux-pipelines service create \
  --name pressure-prediction \
  --template mlflow \
  --model-name battery-pressure-model \
  --model-version 3
```

This generates:

#### Service Code (`pressure-prediction.py`)
```python
import os
import mlflow
from kservice import KService, LoadFrom
from pydantic import BaseModel

class YourModelObject:
    def __init__(self):
        # Load model from MLflow catalog using environment variables
        model_name = os.getenv('MLFLOW_MODEL_NAME', 'battery-pressure-model')
        model_version = os.getenv('MLFLOW_MODEL_VERSION', '3')

        # Construct the model URI for MLflow catalog
        model_uri = f"models:/{model_name}/{model_version}"
        self._model = mlflow.pyfunc.load_model(model_uri)

        # Set is_trained flag on the underlying PXS model (if applicable)
        try:
            if hasattr(self._model, 'unwrap_python_model'):
                unwrapped = self._model.unwrap_python_model()
                if hasattr(unwrapped, 'model'):
                    unwrapped.model.is_trained = True
        except Exception:
            pass  # Continue anyway - the model might still work

    def predict(self, data):
        model_input = [{"data/coolant_surface/points": data}]
        return self._model.predict(model_input)

service = KService(name="pressure-prediction")

@service.predict(...)
def predict(request, model):
    result = model.predict(request.data)
    return response
```

#### Deployment Configuration (`kserve_deploy.yaml`)
```yaml
service_name: pressure-prediction
docker_url: 585768152950.dkr.ecr.eu-west-2.amazonaws.com/pressure-prediction:latest
compute_resources:
  cpu: 1
  memory: 1Gi
  ephemeral_storage: 1G
  gpu:
    count: 1
    name: a10g
envs:
  # MLflow model configuration
  MLFLOW_MODEL_NAME: "battery-pressure-model"
  MLFLOW_MODEL_VERSION: "3"
```

### Updating Model Versions

To deploy a new model version, simply update the environment variables in `kserve_deploy.yaml`:

```yaml
envs:
  MLFLOW_MODEL_NAME: "battery-pressure-model"
  MLFLOW_MODEL_VERSION: "4"  # Updated version
```

Then redeploy the service - no code changes needed!

### 2. Manual Service Creation (Basic Template)

For non-MLflow services, you can create basic services manually:

#### pyproject.toml
          - name: "battery"
            dockerfile: "services/battery/Dockerfile"
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my_new_service"
version = "0.1.0"
description = "Description of your service"
requires-python = "==3.11.*"
dependencies = [
    "physicsx.kservice>=0.1.0",
    # Add model-specific dependencies
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.4.1",
    "ruff>=0.12.8",
]

[[tool.uv.index]]
name = "physicsx"
url = "https://physicsx.jfrog.io/artifactory/api/pypi/px-pypi-release/simple"
```

#### Service Implementation (my_new_service.py)

```python
"""KService implementation for my_new_service."""

from typing import Annotated
from kservice import KService, LoadFrom
from pydantic import BaseModel, Field
import numpy as np

# Define request schema
class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    input_data: list[float] = Field(description="Input features for prediction")
    metadata: dict = Field(default_factory=dict, description="Optional metadata")

# Define response schema
class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: float = Field(description="Model prediction")
    confidence: float = Field(description="Prediction confidence")
    model_version: str = Field(description="Model version used")

# Model class
class MyModel:
    """Your model implementation."""

    def __init__(self):
        # Load your model here (from file, MLflow, etc.)
        self.model_version = "1.0.0"

    def predict(self, input_data: list[float]) -> tuple[float, float]:
        """Make prediction and return confidence."""
        # Your prediction logic here
        prediction = sum(input_data) / len(input_data)  # Example
        confidence = 0.95  # Example
        return prediction, confidence

# Initialize KService
service = KService(name="my_new_service")

@service.load()
def load_model() -> MyModel:
    """Load model when service starts."""
    return MyModel()

@service.predict(request_schema=PredictionRequest, response_schema=PredictionResponse)
def predict(
    request: PredictionRequest,
    model: Annotated[MyModel, LoadFrom(load_model)],
) -> PredictionResponse:
    """Handle prediction requests."""
    prediction, confidence = model.predict(request.input_data)

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        model_version=model.model_version
    )

if __name__ == "__main__":
    service.start()
```

#### Dockerfile

```dockerfile
# Base Python image from PhysicsX registry
FROM 585768152950.dkr.ecr.eu-west-2.amazonaws.com/physicsx/python:3.11-base

# Set NETRC (required for artifactory during uv sync)
ENV NETRC=/run/secrets/.netrc

# Copy project files from services/my_new_service/ (build context is repository root)
COPY services/my_new_service/pyproject.toml services/my_new_service/uv.lock services/my_new_service/my_new_service.py ./

# Install dependencies
RUN --mount=type=secret,id=.netrc,uid=1000 \
    uv sync --frozen
```

#### Deployment Configuration (kserve_deploy.yaml)

```yaml
service_name: my_new_service
docker_url: 585768152950.dkr.ecr.eu-west-2.amazonaws.com/my_new_service:latest
compute_resources:
  cpu: 1
  memory: 1Gi
  ephemeral_storage: 1G
  # Add GPU if needed:
  # gpu:
  #   count: 1
  #   name: a10g
envs:
  # Add environment variables if needed
  # MODEL_PATH: "/app/models"
  # API_KEY: "your-api-key"
```

#### Test File (test_my_new_service.py)

```python
"""Tests for my_new_service."""

from my_new_service import (
    PredictionRequest,
    PredictionResponse,
    MyModel,
    load_model,
    predict,
)

def test_load_model():
    """Test model loading."""
    model = load_model()
    assert isinstance(model, MyModel)
    assert model.model_version == "1.0.0"

def test_predict():
    """Test prediction logic."""
    model = load_model()
    request = PredictionRequest(input_data=[1.0, 2.0, 3.0])
    response = predict(request, model)

    assert isinstance(response, PredictionResponse)
    assert response.prediction == 2.0  # Average of [1,2,3]
    assert response.confidence == 0.95
    assert response.model_version == "1.0.0"
```

### 3. Building Docker Images

#### Add Service to GitHub Actions

After creating a service with `flux-pipelines service create`, copy the generated `image-build.yaml` to .github/workflows.

#### Trigger Build

```bash
# Commit your service
git add services/my_new_service/
git commit -m "Add my_new_service KService implementation"
git push origin your-branch

# Or trigger manually via GitHub Actions UI
# Go to Actions tab -> "Build and Push KService Docker images" -> "Run workflow"
```

### 4. Deploying the Service

The service templates come with a `kserve_deploy.yaml` file that's ready to use. By default, it uses the `:latest` tag which will automatically pull the most recent image built by your GitHub workflow.

#### Optional: Using Specific Image Tags

If you want to use a specific image version instead of `:latest`, you can find the image tag from your GitHub Actions output:

1. Go to your repository's **Actions** tab
2. Find the **"Build and Push KService Docker images"** workflow run
3. Look for your service's build output to get the specific tag

Then update your `kserve_deploy.yaml`:

```yaml
service_name: my_new_service
docker_url: "585768152950.dkr.ecr.eu-west-2.amazonaws.com/my_new_service:specific-tag"
# ... rest of configuration
```

> **💡 Tip**: For most use cases, using `:latest` is sufficient and automatically gets your newest code.

### 5. Deploy Using flux-pipelines

```bash
# Navigate to service directory
cd services/my_new_service

# Deploy the service
flux-pipelines service deploy

# The service will be deployed to the cluster and you'll see output like:
# Service 'my_new_service' deployed successfully
# Endpoint: http://my_new_service.your-tenant.svc.cluster.local
```

### 6. Listing Services

```bash
# List all deployed services
flux-pipelines service list

# Output will show:
# my_new_service @http://my_new_service.your-tenant.svc.cluster.local
```

### 7. Calling Services

#### MLflow Service Example

For an MLflow service (e.g., `pressure-prediction`):

```bash
# Test pressure prediction endpoint
curl -X POST "http://pressure-prediction.your-tenant.svc.cluster.local/v1/models/pressure-prediction:predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1.2, 3.4, 5.6], [7.8, 9.0, 1.1]]
  }'

# Expected response:
# {
#   "prediction": [[pressure_values_for_point_1], [pressure_values_for_point_2]]
# }
```

#### Basic Service Example

For a basic service:

```bash
# Test prediction endpoint
curl -X POST "http://my_basic_service.your-tenant.svc.cluster.local/v1/models/my_basic_service:predict" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello KService!"
  }'

# Expected response:
# {
#   "echo": "Echo: Hello KService!"
# }
```

#### Using Python Client

```python
import httpx

# MLflow service example
mlflow_url = "http://pressure-prediction.your-tenant.svc.cluster.local/v1/models/pressure-prediction:predict"
mlflow_payload = {
    "data": [[1.2, 3.4, 5.6], [7.8, 9.0, 1.1]]  # Point coordinates
}

response = httpx.post(mlflow_url, json=mlflow_payload)
predictions = response.json()
print(f"Pressure predictions: {predictions['prediction']}")

# Basic service example
basic_url = "http://my_basic_service.your-tenant.svc.cluster.local/v1/models/my_basic_service:predict"
basic_payload = {"message": "Hello from Python!"}

response = httpx.post(basic_url, json=basic_payload)
result = response.json()
print(f"Echo response: {result['echo']}")
```

#### Service Health Check

```bash
# Check if service is healthy
curl "http://my_new_service.your-tenant.svc.cluster.local/v1/models/my_new_service"

# Response indicates service status and metadata
```

## Docker Image Optimization

Service templates are optimized for their intended use case:

### 🪶 **Basic Services (CPU-Only)**
- **Base Image**: `ghcr.io/astral-sh/uv:python3.11-bookworm-slim`
- **Size**: ~200MB (much smaller)
- **Use Case**: HTTP endpoints, simple data processing, non-ML services
- **GPU Support**: None (commented out in `kserve_deploy.yaml`)

### 🚀 **MLflow Services (GPU-Enabled)**
- **Base Image**: `585768152950.dkr.ecr.eu-west-2.amazonaws.com/physicsx/python:3.11-cuda-12.6`
- **Size**: ~2GB (larger but optimized for ML)
- **Use Case**: ML model inference, deep learning models
- **GPU Support**: Enabled by default (A10G GPU)

### 🎯 **When to Use Which**

| Service Type          | Use Basic Template | Use MLflow Template |
| --------------------- | ------------------ | ------------------- |
| **API endpoints**     | ✅                  | ❌                   |
| **Data validation**   | ✅                  | ❌                   |
| **Simple processing** | ✅                  | ❌                   |
| **Model inference**   | ❌                  | ✅                   |
| **Deep learning**     | ❌                  | ✅                   |
| **Computer vision**   | ❌                  | ✅                   |

> **💡 Pro Tip**: Always start with the `basic` template unless you specifically need ML inference. You can always rebuild with a GPU image later if needed.

## Best Practices

### Components
- **Version strategically**: Use semantic versioning and bump versions only when necessary
- **Test locally**: Always test components with `debug_component.py` before committing
- **Resource management**: Set appropriate CPU/memory limits in pipelines
- **Documentation**: Update component README with usage examples

### Services
- **Schema validation**: Use Pydantic schemas for robust request/response validation
- **Error handling**: Implement proper error handling in prediction endpoints
- **Resource sizing**: Start with minimal resources and scale based on load testing
- **Monitoring**: Use service logs and metrics to monitor performance
- **Security**: Never expose sensitive data in service responses

### CI/CD
- **Branch protection**: Use pull requests to review changes before merging
- **Testing**: Ensure all tests pass before deployment
- **Rollback**: Keep previous versions available for quick rollback if needed
