import torch.nn as nn


class CMAMLoss(nn.Module):
    def __init__(
        self,
        cosine_weight=1.0,
        mae_weight=1.0,
        mse_weight=1.0,
        maximize_cosine=True,
        epsilon=1e-8,
    ):
        super(CMAMLoss, self).__init__()
        self.cosine_weight = cosine_weight
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.maximize_cosine = maximize_cosine
        self.epsilon = epsilon

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=epsilon)
        self.mae_loss = nn.L1Loss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")

    def __str__(self):
        return f"CMAMLoss(cosine_weight={self.cosine_weight}, mae_weight={self.mae_weight}, mse_weight={self.mse_weight})"

    def forward(self, predictions, targets):
        # Cosine Similarity
        cosine_sim = self.cosine_similarity(predictions, targets).mean()
        cosine_loss = -cosine_sim if self.maximize_cosine else cosine_sim

        # Mean Absolute Error
        mae = self.mae_loss(predictions, targets)

        # Mean Squared Error
        mse = self.mse_loss(predictions, targets)

        # Combine the losses with their respective weights
        total_loss = (
            self.cosine_weight * cosine_loss
            + self.mae_weight * mae
            + self.mse_weight * mse
        )

        # Create a dictionary with individual loss terms
        loss_dict = {
            "total_loss": total_loss,
            "cosine_sim": cosine_sim,
            "mae": mae,
            "mse": mse,
        }

        return loss_dict
