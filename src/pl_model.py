import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from matplotlib import cm
from cifarDataset import CIFAR10_CLASSES
from pl_datamodule import CIFAR_MEAN, CIFAR_STD
from ViT import ViT

class ViTClassifier(pl.LightningModule):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        embedding_dim=128,
        num_heads=4,
        num_layers=6,
        dropout=0.1,
        lr=3e-4,
        weight_decay=0.05,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros(1, 3, image_size, image_size)
        
        self.sample = {"images": [], "labels": [], "attentions": [], "predictions": []}

    def _attention_overlay(self, preview, grid):
        values = grid.squeeze(1).clamp(0, 1).detach().cpu().numpy()
        jet = torch.from_numpy(cm.jet(values)[..., :3]).to(dtype=preview.dtype)
        jet = jet.permute(0, 3, 1, 2)
        return (0.45 * preview + 0.55 * jet).clamp(0, 1)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        images, labels = batch
        logits, attentions = self(images)
        loss = self.criterion(logits, labels)
        predictions = logits.argmax(dim=-1)
        acc = (predictions == labels).float().mean()
        self.log(f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss, attentions, predictions

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx != 0 or self.current_epoch != 0:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_images"):
            return
        images, _ = batch
        mean = images.new_tensor(CIFAR_MEAN)[:, None, None]
        std = images.new_tensor(CIFAR_STD)[:, None, None]
        preview = (images[:16] * std + mean).clamp(0, 1)
        experiment.add_images("train/samples", preview, global_step=self.global_step)

    def training_step(self, batch, batch_idx):
        loss, attentions, predictions = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        loss, attentions, predictions = self._step(batch, "val")
        if batch_idx == 0 and not self.trainer.sanity_checking:
            n = min(16, images.size(0))
            self.sample = {
                "images": images[:n].detach().cpu(),
                "labels": labels[:n].detach().cpu(),
                "predictions": predictions[:n].detach().cpu(),
                "attentions": [a[:n].detach().cpu() for a in attentions],
            }
        return loss

    def on_validation_epoch_end(self):
        experiment = getattr(self.logger, "experiment", None)
        images = self.sample.get("images")
        if (
            experiment is None
            or not hasattr(experiment, "add_images")
            or not torch.is_tensor(images)
        ):
            return

        mean = images.new_tensor(CIFAR_MEAN)[:, None, None]
        std = images.new_tensor(CIFAR_STD)[:, None, None]
        preview = (images * std + mean).clamp(0, 1)

        # attentions[layer]: (B, heads, tokens, tokens)
        grid_size = self.hparams.image_size // self.hparams.patch_size
        attention = torch.stack(self.sample["attentions"]).mean(dim=0)
        cls_to_patches = attention[:, :, 0, 1:]
        grid = cls_to_patches.mean(dim=1).reshape(-1, 1, grid_size, grid_size)
        grid = grid / grid.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        grid = F.interpolate(grid, size=preview.shape[-2:], mode="bilinear", align_corners=False)
        overlay = self._attention_overlay(preview, grid)
        experiment.add_images("val/attention_map", overlay, global_step=self.global_step)

        lines = []
        for i, (pred, label) in enumerate(
            zip(self.sample["predictions"], self.sample["labels"])
        ):
            pred_name = CIFAR10_CLASSES[int(pred)]
            label_name = CIFAR10_CLASSES[int(label)]
            lines.append(f"{i}: pred={pred_name}  label={label_name}")
        experiment.add_text("val/predictions", "\n".join(lines), global_step=self.global_step)

        self.sample = {"images": [], "labels": [], "attentions": [], "predictions": []}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
