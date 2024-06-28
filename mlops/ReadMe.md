# MLOps Environment Setup

This Docker Compose setup creates a local MLOps environment featuring MinIO for artifact storage, PostgreSQL as a database backend for MLflow, MLflow for experiment tracking and model management, Jupyter for interactive development and data analysis, and TFX for end-to-end machine learning pipelines.

## Services Included

- **MinIO**: S3-compatible object storage for ML artifacts.
- **PostgreSQL**: Database for MLflow's backend store.
- **MLflow**: Platform to manage the ML lifecycle, including experimentation, reproducibility, and deployment.
- **Jupyter**: Interactive development environment for exploratory research and analysis.
- **TFX**: Framework for deploying production ML pipelines.

## Prerequisites

- Docker and Docker Compose installed on your machine.
- Basic familiarity with Docker, MLflow, Jupyter, and TFX.

## Configuration

Before running the Docker Compose setup, ensure you have created a `.env` file in the same directory as your `docker-compose.yml` file with the following environment variables:

```env
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
POSTGRES_DB=postgres
POSTGRES_USER=root
POSTGRES_PASSWORD=root
```


Running the Setup
To start the services, run the following command in the directory containing your docker-compose.yml file:
```
docker network create shared_mlops_network
docker-compose --env-file config/.env -f docker-compose.yml up -d --build
docker-compose --env-file deployment/config/.env -f deployment/docker-compose.yml up -d --build
```


## Accessing Services
- MLflow: Navigate to http://localhost:5000 on your web browser.
- MinIO: Access the MinIO console at http://localhost:9001 with the credentials you specified.
- Jupyter: After starting, the Jupyter service will log a URL with a token in the console. Use this URL to access Jupyter Lab.


## Notebooks and Pipelines
Place your Jupyter notebooks in the ./notebooks directory to make them accessible in the Jupyter environment.
Store your TFX pipeline definitions in the ./tfx_pipelines directory for use within the TFX service.
