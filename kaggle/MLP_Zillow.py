# DATA: https://www.kaggle.com/competitions/zillow-prize-1/data

# SOURCES:
# https://www.kaggle.com/competitions/zillow-prize-1/discussion/39578
# https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from pandas.api.types import is_numeric_dtype
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from kaggle.kaggle_models.MLP_Zillow_model import MLP
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os.path


def main():
    # Load datasets
    train = pd.read_csv('kaggle_data/zillow-prize-1/train_2016_v2.csv', parse_dates=["transactiondate"])
    properties = pd.read_csv('kaggle_data/zillow-prize-1/properties_2016.csv')
    test = pd.read_csv('kaggle_data/zillow-prize-1/sample_submission.csv')
    test = test.rename(
        columns={'ParcelId': 'parcelid'})  # To make it easier for merging datasets on same column_id later

    # Merging datasets
    df_train = train.merge(properties, how='left', on='parcelid')
    df_test = test.merge(properties, how='left', on='parcelid')

    # Ensuring datasets have same columns
    y_train = df_train['logerror'].values
    common_cols = list(set(df_train.columns).intersection(df_test.columns))
    df_train = df_train[common_cols]
    df_test = df_test[common_cols]

    # Remove previous variables to keep some memory
    del properties, train
    gc.collect()

    # Cleaning up data
    df_train.fillna(0, inplace=True)
    df_test.fillna(0, inplace=True)
    for column in df_train:
        if not is_numeric_dtype(df_train[column]):
            df_train[column] = pd.factorize(df_train[column])[0]
            df_test[column] = pd.factorize(df_test[column])[0]

    # Scaling data
    X_train = df_train.values # returns a numpy array
    X_test = df_test.values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Creating validation data
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_scaled, y_train, test_size=0.2)

    # Converting data to pytorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_valid, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_valid, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Train model if weight does not exist
    model = MLP(X_train_tensor.shape[1])
    save_path = 'kaggle_weights/MLP_Zillow.pt'
    if not os.path.isfile(save_path):
        # Setup model
        max_epochs = 5
        batch_size = 32
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Train model
        trainer = pl.Trainer(max_epochs=max_epochs)
        trainer.fit(model, train_dataloader, val_dataloader)

        # Get median losses
        loss_train, loss_val = model.train_loss, model.val_loss
        loss_per_train_epoch, loss_per_val_epoch = int(len(loss_train) / max_epochs), int(len(loss_val) / max_epochs)
        loss_train_avg, loss_val_avg = [], []
        for epoch in range(max_epochs):
            start_train_idx, end_train_idx = loss_per_train_epoch * epoch, loss_per_train_epoch * (epoch+1)
            start_val_idx, end_val_idx = loss_per_val_epoch * epoch, loss_per_val_epoch * (epoch + 1)
            loss_train_avg.append(torch.median(torch.FloatTensor(loss_train[start_train_idx:end_train_idx])).cpu().item())
            loss_val_avg.append(torch.median(torch.FloatTensor(loss_val[start_val_idx:end_val_idx])).cpu().item())

        # Graph Loss
        epochs = range(1, max_epochs+1)
        plt.plot(epochs, loss_train_avg, 'g', label='Training loss')
        plt.plot(epochs, loss_val_avg, 'b', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.show()

        torch.save(model.state_dict(), 'kaggle_weights/MLP_Zillow.pt')

    # Predicting the results
    model.load_state_dict(torch.load(save_path))
    with torch.no_grad():
        Predicted_test_MLP = model(X_test_tensor)
    # Submitting the Results
    sample_file = pd.read_csv('kaggle_data/zillow-prize-1/sample_submission.csv')
    for c in sample_file.columns[sample_file.columns != 'ParcelId']:
        sample_file[c] = Predicted_test_MLP
    print('Preparing the csv file …')
    sample_file.to_csv('kaggle_submissions/MLP_Zillow_results.csv', index=False, float_format='%.4f')
    print("Finished writing the file")

if __name__ == "__main__":
    main()