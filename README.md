# Mlflow_Triton
Serving models with Mlfow and Triton Inference Server


For serving models with Triton inference Server, Nvidia provides Mlflow Triton Plugin.

[https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/containers/mlflow-triton-plugin](<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/containers/mlflow-triton-plugin>)

[https://github.com/nv-morpheus/Morpheus/tree/bc791eaec7ffa19db2fd292f8fb65a74473885a2/models/mlflow](<https://github.com/nv-morpheus/Morpheus/tree/bc791eaec7ffa19db2fd292f8fb65a74473885a2/models/mlflow>)

Currently it supports <u>onnx </u>

and <u>tensorRT</u>

 model flavours.

**Steps:**

1. Run Triton Inference Server

2. Run MLFlow Triton Plugin

3. Publish Models to Mlflow server

4. Deploy the published models to Triton



Start Triton Inference Server in <u>EXPLICIT</u>

 mode

```bash
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/ubuntu/triton_models:/models nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models --model-control-mode=explicit

# Explicit mode does not load models at runtime.
```



**Mlfow Triton Plugin**:

Create a folder in the machine with name <u>triton_models</u>

 and copy your models to this folder with model structure as required by Triton Server

**Model Structure for Inferencing:**

```
└── model_folder/      # model_folder
    ├── 1              # Version of the model
        └── model.ckpt # model file
    ├── config.pbxt    # model configfile
    └── labels.txt     # labels of classes
```



Create MLFlow Triton Plugin container with volume mount to Triton model repository and open bash in the container:

```bash
docker run -it -v /home/ubuntu/triton_models:/triton_models \
--env TRITON_MODEL_REPO=/triton_models \
--gpus '"device=0"' \
--net=host \
--rm \
-d nvcr.io/nvidia/morpheus/mlflow-triton-plugin:1.28.0

docker exec -it <container_name> bash
```



Starting the mlflow server:

```bash
nohup mlflow server --backend-store-uri sqlite:////tmp/mlflow-db.sqlite --default-artifact-root /mlflow/artifacts --host 0.0.0.0 &
```

Publish reference models to MLflow:

```bash
python publish_model_to_mlflow.py --model_name densenet_onnx  --model_directory /mlflow/model_repository/densenet_onnx --flavor triton
```



Create Deployments the models to Triton Inference Servre:

```bash
mlflow deployments create -t triton --flavor triton --name densenet_onnx -m models:/densenet_onnx/1

mlflow deployments delete -t triton --name densenet_onnx

mlflow deployments update -t triton --flavor triton --name densenet_onnx -m models:/densenet_onnx/2
```
