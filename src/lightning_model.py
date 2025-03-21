import pytorch_lightning as pl
import torch as th
import timeit
import os
import json
import wandb
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
# from src.utils import NumpyEncoder, calculate_fpr_fnr_with_global, plot_confusion_matrix

import itertools

import matplotlib.pyplot as plt


def calculate_fpr_fnr_with_global(cm):
    """
    Calculate FPR and FNR for each class and globally for a multi-class confusion matrix.

    Parameters:
        cm (numpy.ndarray): Confusion matrix of shape (num_classes, num_classes).

    Returns:
        dict: A dictionary containing per-class and global FPR and FNR.
    """
    num_classes = cm.shape[0]
    results = {"per_class": {}, "global": {}}

    # Initialize variables for global calculation
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0

    # Per-class calculation
    for class_idx in range(num_classes):
        TP = cm[class_idx, class_idx]
        FN = np.sum(cm[class_idx, :]) - TP
        FP = np.sum(cm[:, class_idx]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        # Calculate FPR and FNR for this class
        FPR = FP / (FP + TN) if (FP + TN) != 0 else None
        FNR = FN / (TP + FN) if (TP + FN) != 0 else None

        # Store per-class results
        results["per_class"][class_idx] = {"FPR": FPR, "FNR": FNR}

        # Update global counts
        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

    # Global calculation
    global_FPR = total_FP / \
        (total_FP + total_TN) if (total_FP + total_TN) != 0 else None
    global_FNR = total_FN / \
        (total_FN + total_TP) if (total_FN + total_TP) != 0 else None

    results["global"]["FPR"] = global_FPR
    results["global"]["FNR"] = global_FNR

    return results


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalized=False,
                          file_path=None,
                          show_figure=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalized:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    if file_path:
        plt.savefig(file_path)
    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class LitClassifier(pl.LightningModule):
    def __init__(self, model, criterion, learning_rate, config, model_name, labels_mapping, weight_decay=0, using_wandb=False, multi_class=False, label_col="Label", class_num_col="Class"):
        """
        model:      your neural network (an instance of nn.Module)
        criterion:  loss function
        learning_rate: learning rate for the optimizer
        weight_decay: L2 regularization
        model_name: the name of the model (used to select the correct dataset, etc.)
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.using_wandb = using_wandb
        self.model_name = model_name
        self.labels = list(labels_mapping.values())
        self.labels_mapping = labels_mapping
        self.multi_class = multi_class
        self.label_col = label_col
        self.class_num_col = class_num_col
        self.save_hyperparameters(config)
        self.train_epoch_metrics = {}
        self.val_epoch_metrics = {}
        self.train_outputs = {"preds": [], "targets": []}
        self.val_outputs = {"preds": [], "targets": []}
        self.test_outputs = {"preds": [], "targets": []}
        self.test_prefix = ""

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print(f"================>> training_step")
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        pred = pred.argmax(dim=1)
        acc = (pred == y).float().mean() * 100.0
        self.log('train_loss', loss, on_epoch=True,
                 prog_bar=True, batch_size=len(x))
        print(f"==>> len(x): {len(x)}")
        self.log('train_acc', acc, on_epoch=True,
                 prog_bar=True, batch_size=len(x))

        self.train_outputs["preds"].append(pred)
        self.train_outputs["targets"].append(y)

        return loss

    def on_train_epoch_end(self):
        print(f"================>> on_train_epoch_end")
        all_preds = th.cat(self.train_outputs["preds"]).detach().cpu().numpy()
        all_targets = th.cat(
            self.train_outputs["targets"]).detach().cpu().numpy()
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100.0
        self.log("train_f1_score", weighted_f1, on_epoch=True,
                 prog_bar=True)

        self.train_outputs = {"preds": [], "targets": []}

    def validation_step(self, batch, batch_idx):
        print(f"================>> validation_step")
        x, y = batch
        print(f"==>> len(x): {len(x)}")
        pred = self(x)
        loss = self.criterion(pred, y)
        pred = pred.argmax(dim=1)
        acc = (pred == y).float().mean() * 100.0
        self.log('val_loss', loss, on_epoch=True,
                 prog_bar=True, batch_size=len(x))
        self.log('val_acc', acc, on_epoch=True,
                 prog_bar=True, batch_size=len(x))

        self.val_outputs["preds"].append(pred)
        self.val_outputs["targets"].append(y)

        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        print(f"================>> on_validation_epoch_end")
        all_preds = th.cat(self.val_outputs["preds"]).detach().cpu().numpy()
        all_targets = th.cat(
            self.val_outputs["targets"]).detach().cpu().numpy()
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100.0
        print(f"==>> weighted_f1: {weighted_f1}")
        self.log("val_f1_score", weighted_f1, on_epoch=True,
                 prog_bar=True)

        self.val_outputs = {"preds": [], "targets": []}

    def test_step(self, batch, batch_idx):
        print(f"================>> test_step")
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        start_time = timeit.default_timer()
        pred = pred.argmax(dim=1)
        elapsed = timeit.default_timer() - start_time
        print(f"==>> elapsed: {elapsed}")

        acc = (pred == y).float().mean() * 100.0

        self.log(f"{self.test_prefix}_test_loss", loss, on_epoch=True,
                 prog_bar=True, batch_size=len(x))
        self.log(f"{self.test_prefix}_test_acc", acc, on_epoch=True,
                 prog_bar=True, batch_size=len(x))
        self.log(f"{self.test_prefix}_elapsed", elapsed, on_epoch=True,
                 prog_bar=True, batch_size=len(x))

        self.test_outputs["preds"].append(pred)
        self.test_outputs["targets"].append(y)
        return {"test_loss": loss, "test_acc": acc, "preds": pred, "targets": y, "elapsed": elapsed}

    def on_test_epoch_end(self):
        print(f"================>> on_test_epoch_end")
        all_preds = th.cat(self.test_outputs["preds"]).detach().cpu().numpy()
        all_targets = th.cat(
            self.test_outputs["targets"]).detach().cpu().numpy()
        self.test_outputs = {"preds": [], "targets": []}
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100.0
        self.log(f"{self.test_prefix}_test_f1", weighted_f1, on_epoch=True,
                 prog_bar=True)

        all_targets = np.vectorize(self.labels_mapping.get)(all_targets)
        all_preds = np.vectorize(self.labels_mapping.get)(all_preds)

        cm = confusion_matrix(all_targets, all_preds, labels=self.labels)

        cr = classification_report(
            all_targets, all_preds, digits=4, output_dict=True, zero_division=0)
        report = classification_report(
            all_targets, all_preds, digits=4, output_dict=False, zero_division=0)
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100

        results_fpr_fnr = calculate_fpr_fnr_with_global(cm)
        fpr = results_fpr_fnr["global"]["FPR"]
        fnr = results_fpr_fnr["global"]["FNR"]

        results = {
            "test_weighted_f1": weighted_f1,
            "test_fpr": fpr,
            "test_fnr": fnr,
            "classification_report": cr,
            "results_fpr_fnr": results_fpr_fnr
        }

        os.makedirs("temp", exist_ok=True)
        json_path = os.path.join("temp", f"{self.model_name}_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

        if self.using_wandb:
            wandb.save(json_path)

        print("=== Test Evaluation Metrics ===")
        print("Classification Report:\n", report)

        cm_normalized = confusion_matrix(
            all_targets, all_preds, labels=self.labels, normalize="true")
        fig = plot_confusion_matrix(cm=cm,
                                    normalized=False,
                                    target_names=self.labels,
                                    title=f"Confusion Matrix of {self.model_name}",
                                    file_path=None,
                                    show_figure=False)

        if self.using_wandb:
            wandb.log({f"confusion_matrix_{self.model_name}": wandb.Image(
                fig), "epoch": self.current_epoch})
        fig = plot_confusion_matrix(cm=cm_normalized,
                                    normalized=True,
                                    target_names=self.labels,
                                    title=f"Confusion Matrix of {self.model_name}",
                                    file_path=None,
                                    show_figure=False)

        if self.using_wandb:
            wandb.log({f"confusion_matrix_{self.model_name}_normalized": wandb.Image(
                fig), "epoch": self.current_epoch})

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.model.parameters(),
                                  lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        return optimizer
