# CI/CD Pipeline Report

## 1. Trigger
The pipeline is triggered on two events:
- Push to the 'main' branch
- Manual trigger via workflow_dispatch

## 2. Environment
- Runs on: Ubuntu latest version

## 3. Stages

### a. Setup
- Checkout code using actions/checkout@v2
- Set up Python 3.8 using actions/setup-python@v2

### b. Dependency Installation
- Upgrade pip to the latest version
- Install project dependencies from requirements.txt

### c. Code Quality Check
- Install flake8
- Run flake8 to check for code style and quality issues

### d. Testing
- Install pytest
- Run tests located in the tests/ directory

### e. Model Training
- Execute the training script (src/train_m1.py)

### f. Artifact Storage
- Archive the trained model (models/iris_model.joblib) as an artifact

### g. Docker Image Building
- Build a Docker image named 'mlops-assignment' using the project's Dockerfile

### h. Docker Hub Authentication
- Log in to Docker Hub using credentials stored in GitHub Secrets

### i. Image Publishing
- Tag the built image with the Docker Hub username
- Push the tagged image to Docker Hub

## 4. Security Considerations
- Docker Hub credentials are stored as GitHub Secrets (DOCKER_USERNAME and DOCKER_PASSWORD)
- Secrets are accessed securely within the workflow

## 5. Continuous Integration Aspects
- Code quality check with flake8
- Automated testing with pytest
- Model training on each push or manual trigger

## 6. Continuous Delivery/Deployment Aspects
- Automated Docker image building
- Automatic pushing of the image to Docker Hub, making it ready for deployment

## 7. Artifacts
- The trained model is saved as a workflow artifact, allowing easy access and download

This pipeline ensures that for each push to the main branch or manual trigger:
1. The code is style-checked
2. Tests are run
3. The model is retrained
4. A new Docker image is built and pushed to Docker Hub

This setup allows for rapid iteration, consistent code quality, and easy deployment of the latest version of the application.