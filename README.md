# Wandb 공부

Wandb는 강력한 MLOps Tool이다. 딥러닝을 학습하면서 항상 써왔지만 Wandb를 깊게 사용해본적이 없고 단순히 log만 찍는데에 사용하였다.  
Wandb의 다양한 기능을 공부하고 기록하기 위해 이 Repository를 만들게 되었다.  

Wandb (Public) : https://wandb.ai/tony9402/wandb-practice

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

## Python Library

```
$ pip3 install -r requirements.txt
```

## Wandb

```
$ wandb login #https://wandb.ai/authorize 에 있는 Token 값 입력
```
