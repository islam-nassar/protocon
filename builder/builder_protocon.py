import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import random


class ProtoConModel(nn.Module):
    """
    Build a ProtoCon model
    """

    def __init__(self, base_encoder, num_classes, dim=64):
        super(ProtoConModel, self).__init__()
        # create the encoder with projection head
        self.encoder = base_encoder(num_classes=dim)
        dim_mlp = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                       nn.LeakyReLU(inplace=True, negative_slope=0.1),
                                       nn.Linear(dim_mlp, dim))
        # self.projector = nn.Linear(dim_mlp, dim)
        self.classifier = nn.Linear(dim_mlp, num_classes)

    def forward(self, images):
        out = self.encoder(images)
        # obtain cluster projections
        proj = self.projector(out)
        proj = F.normalize(proj, dim=1)
        # obtain class predictions
        preds = self.classifier(out)
        return preds, proj