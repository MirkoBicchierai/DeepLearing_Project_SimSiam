Instruction to reproduce the experimental results:

1: Setup a Virtual Environment

    Navigate to the project directory, create and activate a virtual environment, then install the required dependencies by running:
    pip install -r requirements.txt

2: Setup MiniImage Dataset

    Download the dataset from the following link:
    https://www.kaggle.com/datasets/arjunashok33/miniimagenet

    Extract the contents of the zip file into the Dataset/CLEAR folder.

    Execute the script to split the dataset:
    python3 split_dataset.py

3: Train the SimSiam Model
    Use the following command to train the SimSiam model:

    python3 main.py [--train_dir] [--val_dir] [--batch_size] [--num_workers] [--base_lr] [--momentum] [--weight_decay] [--epochs]
                    [--val_step] [--knn_k] [--knn_t] [--projector_dim] [--predictor_dim]

    --train_dir (default: 'Dataset/SPLITTED/Train')
    Path to the training dataset directory.

    --val_dir (default: 'Dataset/SPLITTED/Test')
    Path to the validation dataset directory.

    --batch_size (default: 96)
    Batch size for training.

    --num_workers (default: 12)
    Number of workers for data loading.

    --base_lr (default: 0.05)
    Base learning rate.

    --momentum (default: 0.9)
    Momentum for the SGD optimizer.

    --weight_decay (default: 0.0001)
    Weight decay for the optimizer.

    --epochs (default: 200)
    Number of training epochs.

    --val_step (default: 1)
    Frequency of validation steps (measured in epochs).

    --knn_k (default: 200)
    Number of nearest neighbors to consider during KNN evaluation.

    --knn_t (default: 0.1)
    Temperature parameter used in KNN evaluation.

    --projector_dim (default: 512)
    Dimensionality of the projector's feature representation.

    --predictor_dim (default: 128)
    Dimensionality of the predictor's feature representation.

    --stop_grad (default: True)
    Whether to stop gradient backpropagation.

    --type_loss (default: 'Cosine Similarity')
    Type of loss function to use. Choices are: ["Cosine Similarity", "Cross Entropy Similarity"].

    --symmetric (default: True)
    Whether to use a symmetric loss.

    Once training is complete, the model will be saved in the Models/SimSiam directory (the scripts save a checkpoint every 10 epoch).


4: Run a linear evaluation
    Perform a linear evaluation using the following command:

    python3 LinearEvaluation.py [--path] [--batch_size] [--epochs 100] [--lr] [--momentum] [--weight_decay] [--num_workers] [--dataset MiniImageNet]
                                [--projector_dim] [--predictor_dim]

    --path (default: 'Models/SimSiam/Symmetric Loss/model_200_96_Final.pth')
    Path to the pretrained model file (Use the path to model before pretrained saved in Models/SimSiam directory or a
    pretrained model inside one of this 'Models/SimSiam/Asymmetric Loss/model_200_96_Final.pth', 'Models/SimSiam/Symmetric Loss/model_200_96_Final.pth',
     'Models/SimSiam/Cross Entropy Similarity/model_200_96_Final.pth', 'Models/SimSiam/Symmetric Loss without stopgrad/model_200_96_Final.pth').

    --batch_size (default: 128)
    Batch size for training.

    --epochs (default: 100)
    Number of epochs for training.

    --lr (default: 0.3)
    Learning rate for the optimizer. Suggested values: 0.025 for Cifar10, 0.3 for MiniImageNet.

    --momentum (default: 0.9)
    Momentum value for the optimizer.

    --weight_decay (default: 0.0)
    Weight decay (regularization) for the optimizer.

    --num_workers (default: 12)
    Number of workers for data loading.

    --dataset (default: 'MiniImageNet')
    Dataset to use in linear evaluation. Choices: ["Cifar10", "MiniImageNet"].

    --projector_dim (default: 512)
    Dimensionality of the projector in the pretrained model (leave the default value if use a pretrained model just saved).

    --predictor_dim (default: 128)
    Dimensionality of the predictor in the pretrained model (leave the default value if use a pretrained model just saved).

    To --projector_dim and --predictor_dim use the same value used for the pretrained SimSiam model


All Experiment are live on Comet_Ml at : https://www.comet.com/mirkobicchierai/deep-learning-project/view/new/panels