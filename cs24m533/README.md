# VGG6 CIFAR-10 experiments (CS6886W Assignment 1)
# Sanjeev Kumar (cs24m533)

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Single- single Runs
# i did this to time the execution during multiple nights
```bash
python train.py --activation relu --optimizer sgd --lr 0.01 --batch_size 128 --epochs 30
python train.py --activation silu --optimizer sgd --lr 0.01 --batch_size 128 --epochs 30
python train.py --activation gelu --optimizer sgd --lr 0.01 --batch_size 128 --epochs 30
python train.py --activation tanh --optimizer sgd --lr 0.01 --batch_size 128 --epochs 30

python train.py --activation relu --optimizer nesterov --lr 0.1 --batch_size 64 --epochs 30
python train.py --activation silu --optimizer nesterov --lr 0.1 --batch_size 64 --epochs 30
python train.py --activation gelu --optimizer nesterov --lr 0.1 --batch_size 64 --epochs 30
python train.py --activation tanh --optimizer nesterov --lr 0.1 --batch_size 64 --epochs 30

python train.py --activation relu --optimizer adam --lr 0.001 --batch_size 128 --epochs 30
python train.py --activation silu --optimizer adam --lr 0.001 --batch_size 128 --epochs 10
python train.py --activation gelu --optimizer adam --lr 0.001 --batch_size 128 --epochs 10
python train.py --activation tanh --optimizer adam --lr 0.001 --batch_size 128 --epochs 10

python train.py --activation relu --optimizer rmsprop --lr 0.01 --batch_size 64 --epochs 10
python train.py --activation silu --optimizer rmsprop --lr 0.01 --batch_size 64 --epochs 20
python train.py --activation gelu --optimizer rmsprop --lr 0.01 --batch_size 64 --epochs 20
python train.py --activation tanh --optimizer rmsprop --lr 0.01 --batch_size 64 --epochs 20

python train.py --activation relu --optimizer sgd --lr 0.1 --batch_size 64 --epochs 20
python train.py --activation silu --optimizer sgd --lr 0.1 --batch_size 64 --epochs 30
python train.py --activation gelu --optimizer sgd --lr 0.1 --batch_size 64 --epochs 30
python train.py --activation tanh --optimizer sgd --lr 0.1 --batch_size 64 --epochs 30

python train.py --activation relu --optimizer adagrad --lr 0.01 --batch_size 64 --epochs 10
python train.py --activation silu --optimizer adagrad --lr 0.01 --batch_size 64 --epochs 20
python train.py --activation gelu --optimizer adagrad --lr 0.01 --batch_size 64 --epochs 20
python train.py --activation tanh --optimizer adagrad --lr 0.01 --batch_size 64 --epochs 20

python train.py --activation sigmoide --optimizer sgd --lr 0.001 --batch_size 64 --epochs 30
python train.py --activation sigmoide --optimizer nesterov --lr 0.1 --batch_size 128 --epochs 30
python train.py --activation sigmoide --optimizer adam --lr 0.1 --batch_size 64 --epochs 10
python train.py --activation sigmoide --optimizer rmsprop --lr 0.01 --batch_size 128 --epochs 20
python train.py --activation sigmoide --optimizer adagrad --lr 0.001 --batch_size 64 --epochs 20
python train.py --activation sigmoide --optimizer adam --lr 0.001 --batch_size 128 --epochs 30


python train.py --activation relu --optimizer nadam --lr 0.001 --batch_size 64 --epochs 10
python train.py --activation silu --optimizer nadam --lr 0.01 --batch_size 128 --epochs 20
python train.py --activation gelu --optimizer nadam --lr 0.1 --batch_size 64 --epochs 30
python train.py --activation tanh --optimizer nadam --lr 0.01 --batch_size 128 --epochs 20
python train.py --activation sigmoide --optimizer nadam --lr 0.01 --batch_size 64 --epochs 30
```

## WandB Sweep
# we can use this for batch run
```bash
wandb login
wandb sweep sweep.yaml
wandb agent <SWEEP_ID>
```

## Output
Models in ./checkpoints, plots and metrics in W&B.
