# Wandb 공부

## 개발환경

- Docker version 20.10.12, build e91ed57
- Docker Image : nvcr.io/nvidia/pytorch:22.05-py3
- Python Version : 3.8.13
- Wandb Version : 0.12.21

## Docker

```
$ docker run -itd --name wandb_practice --net=host --ipc=host --gpus all -v /host/directory:/docker/directory nvcr.io/nvidia/pytorch:22.05-py3
$ docker attach wandb_practice
```

## Wandb

```
$ pip3 install wandb==0.12.21 # Install Wandb
$ wandb login #https://wandb.ai/authorize 에 있는 Token 값 입력
```
