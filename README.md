# SimSiam Implementation and Linear Evaluation

A PyTorch implementation of SimSiam for self-supervised learning, with comprehensive evaluation on MiniImageNet and CIFAR-10 datasets.

## 🚀 Quick Start

1. Clone the repository
2. Set up the environment
3. Download the dataset
4. Train the model
5. Run evaluation

## 📋 Requirements

```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset Setup

### MiniImageNet Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet)
2. Extract the contents into `Dataset/CLEAR` folder
3. Run the dataset splitting script:
   ```bash
   python3 split_dataset.py
   ```

## 🔧 Training SimSiam

Train the model using the following command:

```bash
python3 main.py [OPTIONS]
```

### Training Parameters

| Parameter | Default Value            | Description |
|-----------|--------------------------|-------------|
| `--key`           | '' | CometML Api Key          |
| `--train_dir` | 'Dataset/Splitted/Train' | Training dataset directory |
| `--val_dir` | 'Dataset/Splitted/Test'  | Validation dataset directory |
| `--batch_size` | 96                       | Training batch size |
| `--num_workers` | 12                       | Number of data loading workers |
| `--base_lr` | 0.05                     | Base learning rate |
| `--momentum` | 0.9                      | SGD momentum |
| `--weight_decay` | 0.0001                   | Weight decay |
| `--epochs` | 200                      | Number of training epochs |
| `--val_step` | 1                        | Validation frequency (epochs) |
| `--knn_k` | 200                      | Number of nearest neighbors for KNN |
| `--knn_t` | 0.1                      | KNN temperature parameter |
| `--projector_dim` | 512                      | Projector dimensionality |
| `--predictor_dim` | 128                      | Predictor dimensionality |
| `--stop_grad` | True                     | Stop gradient backpropagation |
| `--type_loss` | 'Cosine Similarity'      | Loss function type |
| `--symmetric` | True                     | Use symmetric loss |

### Available Loss Functions
- "Cosine Similarity"
- "Cross Entropy Similarity"

### Model Checkpoints
The model saves checkpoints every 10 epochs in the `Models/SimSiam` directory.

## 📈 Linear Evaluation

Evaluate the trained model using:

```bash
python3 LinearEvaluation.py [OPTIONS]
```

### Evaluation Parameters

| Parameter         | Default Value | Description              |
|-------------------|---------------|--------------------------|
| `--key`           | '' | CometML Api Key          |
| `--path`          | 'Models/SimSiam/Symmetric Loss/model_200_96_Final.pth' | Pretrained model path    |
| `--batch_size`    | 128 | Evaluation batch size    |
| `--epochs`        | 100 | Evaluation epochs        |
| `--lr`            | 0.3 | Learning rate            |
| `--momentum`      | 0.9 | Optimizer momentum       |
| `--weight_decay`  | 0.0 | Weight decay             |
| `--num_workers`   | 12 | Data loading workers     |
| `--dataset`       | 'MiniImageNet' | Evaluation dataset       |
| `--projector_dim` | 512 | Projector dimensionality |
| `--predictor_dim` | 128 | Predictor dimensionality |

### Dataset Options
- MiniImageNet (default)
- CIFAR-10

### Learning Rate Recommendations
- CIFAR-10: 0.025
- MiniImageNet: 0.3

## 🔍 Pretrained Models

Available model variants:
- `Models/SimSiam/Symmetric Loss/model_200_96_Final.pth`
- `Models/SimSiam/Asymmetric Loss/model_200_96_Final.pth`
- `Models/SimSiam/Cross Entropy Similarity/model_200_96_Final.pth`
- `Models/SimSiam/Symmetric Loss without stopgrad/model_200_96_Final.pth`

## 📊 Experiment Results

All experimental results are tracked and available on Comet.ml:
[View Results Dashboard](https://www.comet.com/mirkobicchierai/deep-learning-project/view/new/panels)

## ⚠️ Important Notes

1. When using pretrained models, ensure `projector_dim` and `predictor_dim` match the training values
2. Checkpoints are saved every 10 epochs
3. Different loss function implementations are available in separate directories
