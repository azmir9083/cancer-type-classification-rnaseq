"""The five models compared in the paper.

Neural models (PyTorch): MLP, 1D-CNN, Autoencoder + softmax.
Classical baselines (scikit-learn): Random Forest, Linear SVM.

All neural models use Adam, dropout and batch normalization. Exact layer sizes,
learning rate and epochs are not given in the paper; the values below are
reasonable defaults chosen for this reimplementation.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    """Two hidden layers with batch normalization and dropout."""

    def __init__(self, n_in, n_classes, h1=512, h2=128, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Treats the standardized gene vector as a 1D signal: strided convolutions and pooling."""

    def __init__(self, n_in, n_classes, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=4, padding=4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(32 * 32, 64),
                                  nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))


class AutoencoderClassifier(nn.Module):
    """Compresses the profile to a 64-value code and classifies from that code."""

    def __init__(self, n_in, n_classes, hidden=512, code=64, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, code), nn.BatchNorm1d(code), nn.ReLU(),
        )
        self.decoder = nn.Sequential(nn.Linear(code, hidden), nn.ReLU(), nn.Linear(hidden, n_in))
        self.classifier = nn.Linear(code, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z), self.decoder(z)


class TorchClassifier:
    """Small scikit-learn-style wrapper so neural and classical models share one interface."""

    def __init__(self, kind, epochs=40, lr=1e-3, batch_size=64, recon_weight=0.5, seed=0):
        self.kind, self.epochs, self.lr = kind, epochs, lr
        self.batch_size, self.recon_weight, self.seed = batch_size, recon_weight, seed

    def _build(self, n_in, n_classes):
        return {"mlp": MLP, "cnn": CNN1D, "ae": AutoencoderClassifier}[self.kind](n_in, n_classes)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        n_classes = int(y.max()) + 1
        self.model = self._build(X.shape[1], n_classes).to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
        n = len(Xt)
        bs = min(self.batch_size, n)
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:  # batch norm needs at least 2 samples
                    continue
                xb, yb = Xt[idx], yt[idx]
                if self.kind == "ae":
                    logits, recon = self.model(xb)
                    loss = ce(logits, yb) + self.recon_weight * mse(recon, xb)
                else:
                    loss = ce(self.model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        out = self.model(Xt)
        logits = out[0] if self.kind == "ae" else out
        return logits.argmax(1).cpu().numpy()


def make_model(name, seed=0, epochs=40):
    """Factory for the five models used in the paper."""
    if name == "Linear SVM":
        return LinearSVC(C=0.01, max_iter=5000, random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    if name == "MLP":
        return TorchClassifier("mlp", epochs=epochs, seed=seed)
    if name == "1D-CNN":
        return TorchClassifier("cnn", epochs=epochs, seed=seed)
    if name == "Autoencoder + softmax":
        return TorchClassifier("ae", epochs=epochs, seed=seed)
    raise ValueError(name)


MODEL_NAMES = ["Linear SVM", "MLP", "Random Forest", "Autoencoder + softmax", "1D-CNN"]
