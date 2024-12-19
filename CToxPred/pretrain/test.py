import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd
from CToxPred.utils import compute_fingerprint_features, compute_descriptor_features, compute_metrics
from CToxPred.pairwise_correlation import CorrelationThreshold

class Nav15Classifier(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(Nav15Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, 400, bias=True)
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        self.bn1 = torch.nn.BatchNorm1d(num_features=400)

        self.linear2 = torch.nn.Linear(400, 200, bias=True)
        torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        self.bn2 = torch.nn.BatchNorm1d(num_features=200)

        self.linear3 = torch.nn.Linear(200, outputSize, bias=True)
        torch.nn.init.kaiming_normal_(self.linear3.weight, nonlinearity='relu')

    def forward(self, x):
        out = torch.relu(self.bn1(self.linear1(x)))
        out = torch.relu(self.bn2(self.linear2(out)))
        out = torch.softmax(self.linear3(out), dim=1)
        return out

    def save(self, path):
        print(f"Saving model... {path}")
        torch.save(self.state_dict(), path)

    def load(self, path):
        device = torch.device('cpu')
        self.load_state_dict(torch.load(path, map_location=device))


def load_data(file_path):
    """
    Load data from a text file.

    Parameters
    ----------
    file_path : str
        Path to the file (train.txt or test.txt).

    Returns
    -------
    smiles_list : List[str]
        List of SMILES strings.
    labels : np.ndarray
        Corresponding labels as a NumPy array.
    """
    data = pd.read_csv(file_path, sep='\t', header=None, names=["smiles", "label"])
    smiles_list = data["smiles"].tolist()
    labels = data["label"].to_numpy(dtype=np.int32)
    return smiles_list, labels


def train_and_evaluate(train_file, test_file, model_save_path):
    # Load train and test data
    print(">>>>>>> Load Data <<<<<<<")
    train_smiles, train_labels = load_data(train_file)
    test_smiles, test_labels = load_data(test_file)

    # Compute features
    print(">>>>>>> Calculate Features <<<<<<<")
    train_fingerprints = compute_fingerprint_features(train_smiles)
    train_descriptors = compute_descriptor_features(train_smiles)
    test_fingerprints = compute_fingerprint_features(test_smiles)
    test_descriptors = compute_descriptor_features(test_smiles)

    # Preprocess descriptors
    print(">>>>>>> Preprocess Descriptors <<<<<<<")
    path = ['..', 'CToxPred', 'models', 'decriptors_preprocessing', 'Nav1.5', 'nav_descriptors_preprocessing_pipeline.sav']
    descriptors_transformation_pipeline = joblib.load(os.path.join(*path))
    train_descriptors = descriptors_transformation_pipeline.transform(train_descriptors)
    test_descriptors = descriptors_transformation_pipeline.transform(test_descriptors)

    # Combine features
    train_features = np.concatenate((train_fingerprints, train_descriptors), axis=1)
    test_features = np.concatenate((test_fingerprints, test_descriptors), axis=1)

    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32),
                                   torch.tensor(train_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32),
                                  torch.tensor(test_labels, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, loss, optimizer
    model = Nav15Classifier(inputSize=train_features.shape[1], outputSize=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    print(">>>>>>> Training Nav1.5 Model <<<<<<<")
    best_accuracy = 0.0
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save(model_save_path)

    print(f"Training complete. Best model saved with accuracy: {best_accuracy:.4f}")


# Example Usage
if __name__ == "__main__":
    train_file = r"C:\Users\ADS_Lab\Desktop\JH\Paper_Replication\CToxPred\dataset\hERG_train_no_salt.txt"
    test_file = r"C:\Users\ADS_Lab\Desktop\JH\Paper_Replication\CToxPred\dataset\hERG_test_no_salt.txt"

    model_save_path = "nav15_best_model.pth"
    train_and_evaluate(train_file, test_file, model_save_path)
