import torch
import torch.nn as nn
import torch.optim as optim


class SpectralGCN(nn.Module):
    """
    Simple Spectral Graph Convolution Network using Fourier filtering.
    """

    def __init__(self, num_nodes, lr=0.05, device=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Learnable spectral filter
        self.filter = nn.Parameter(torch.randn(num_nodes, 1))
        # self.filter_raw = nn.Parameter(torch.randn(num_nodes, 1))
        # self.filter = torch.sigmoid(self.filter_raw)

        # Optimizer + loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.to(self.device)

    # ------------------------------------------------------------
    # Forward pass (spectral convolution)
    # ------------------------------------------------------------
    def forward(self, x, U):
        """
        x : (N, 1) signal on nodes
        U : (N, N) eigenvectors of Laplacian
        """
        x = x.to(self.device)
        U = U.to(self.device)

        # Fourier transform
        x_hat = U.T @ x

        # Apply learnable spectral filter (sigmoid to stabilise)
        x_hat_filtered = x_hat * torch.sigmoid(self.filter)

        # Inverse Fourier transform
        return U @ x_hat_filtered

    # ------------------------------------------------------------
    # Training with energy conservation term
    # ------------------------------------------------------------
    def fit_conservation(self, X_heat, U, epochs=300, print_every=50, lambda_energy=1.0):
        """
        Autoregressive training with additional energy conservation constraint.
        """

        X_heat = X_heat.to(self.device)
        U = U.to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss = 0

            # Autoregressive one-step prediction
            for t in range(X_heat.shape[1] - 1):
                x_pred = self.forward(X_heat[:, t:t + 1], U)
                x_true = X_heat[:, t + 1:t + 2]

                # MSE loss
                mse = self.loss_fn(x_pred, x_true)

                # Energy (sum of heat) preservation loss
                energy_pred = x_pred.sum()
                energy_true = x_true.sum()
                energy_loss = (energy_pred - energy_true).pow(2)

                total_loss += mse + lambda_energy * energy_loss

            total_loss.backward()
            self.optimizer.step()

            if epoch % print_every == 0:
                print(f"[Epoch {epoch}] Loss: {total_loss.item():.6f}")

        return self

    # ------------------------------------------------------------
    # Teacher-forced one-step training (most stable)
    # ------------------------------------------------------------
    def fit(self, X_heat, U, epochs=300, print_every=50):
        """
        Standard autoregressive training:
        Learn x_{t+1} = f(x_t).
        """

        X_heat = X_heat.to(self.device)
        U = U.to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = 0.0

            # One-step predictions across the whole sequence
            for t in range(X_heat.shape[1] - 1):
                x_t = X_heat[:, t:t + 1]
                x_next = X_heat[:, t + 1:t + 2]

                x_pred = self.forward(x_t, U)
                loss += self.loss_fn(x_pred, x_next)

            loss.backward()
            self.optimizer.step()

            if epoch % print_every == 0:
                print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

        return self

    # ------------------------------------------------------------
    # Optional helper
    # ------------------------------------------------------------
    @staticmethod
    def enforce_energy(pred, target):
        """Rescale prediction to match total energy."""
        return pred * (target.sum() / pred.sum())

    # ------------------------------------------------------------
    # Predict full sequence autoregressively
    # ------------------------------------------------------------
    def predict(
        self,
        X_heat,
        U,
        steps=None,
        enforce_energy=False,
        target_energy_seq=None
    ):
        """
        Predict future heat using the learned model.
        Supports optional energy conservation.
        """

        X_heat = X_heat.to(self.device)
        U = U.to(self.device)

        T = steps or X_heat.shape[1]

        # Initial state
        x = X_heat[:, 0:1]
        preds = [x]

        # Optional: target energy over time
        if enforce_energy and target_energy_seq is None:
            target_energy_seq = X_heat.sum(dim=0)

        with torch.no_grad():
            for t in range(T - 1):
                x_pred = self.forward(x, U)

                if enforce_energy:
                    # Match total energy
                    E_pred = x_pred.sum()
                    E_target = (
                        target_energy_seq[t + 1]
                        if target_energy_seq is not None
                        else x.sum()
                    )

                    if E_pred.abs() > 1e-10:
                        x_pred = x_pred * (E_target / E_pred)

                preds.append(x_pred)
                x = x_pred

        return torch.cat(preds, dim=1)

    # ------------------------------------------------------------
    # Simpler predict (no energy conservation)
    # ------------------------------------------------------------
    # def predict(self, X_heat, U, steps=None):
    #     """
    #     Basic autoregressive forward prediction.
    #     """
    #
    #     X_heat = X_heat.to(self.device)
    #     U = U.to(self.device)
    #
    #     T = steps or X_heat.shape[1]
    #
    #     x = X_heat[:, 0:1]
    #     preds = [x]
    #
    #     with torch.no_grad():
    #         for _ in range(T - 1):
    #             x = self.forward(x, U)
    #             preds.append(x)
    #
    #     return torch.cat(preds, dim=1)
