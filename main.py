import torch.nn as nn
import os
import pickle
import torch
import warnings
import wandb
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from src.models import NIDSFNN, NIDSCNN, NIDSGRU, NIDSLSTM, CNN_LSTM
from src.lightning_model import LitClassifier
from src.lightning_data import LitDataModule
from src.dataset.dataset_info import datasets, network_features
from local_variables import local_datasets_path

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def main():
    using_wandb = True
    multi_class = True

    sort_timestamp = False
    sort_after_partition = False

    use_port_in_address = False
    generated_ips = False

    # Hyperparameters
    # dataset_name = "cic_ton_iot_5_percent"
    # dataset_name = "cic_ton_iot"
    # dataset_name = "cic_ids_2017_5_percent"
    # dataset_name = "cic_ids_2017"
    dataset_name = "cic_bot_iot"
    # dataset_name = "cic_ton_iot_modified"
    # dataset_name = "nf_ton_iotv2_modified"
    # dataset_name = "ccd_inid_modified"
    # dataset_name = "nf_uq_nids_modified"
    # dataset_name = "edge_iiot"
    # dataset_name = "nf_cse_cic_ids2018"
    # dataset_name = "nf_bot_iotv2"
    # dataset_name = "nf_uq_nids"
    # dataset_name = "x_iiot"

    early_stopping_patience = max_epochs = 5
    # early_stopping_patience = 30
    batch_size = 128
    learning_rate = 0.001
    weight_decay = 0.001
    fnn_hidden_units = [100, 200]
    cnn_out_channels_list = [64]
    rnn_num_layers = 2
    rnn_hidden_size = 80
    dropout = 0.3
    sequence_length = 1
    stride = 1
    activation = F.relu
    multi_class = True

    exp_type = "gdlc"

    if multi_class:
        exp_type += "__multi_class"

    if use_port_in_address:
        exp_type += "__ports"

    if generated_ips:
        exp_type += "__generated_ips"

    if sort_timestamp:
        exp_type += "__sorted"
    elif sort_after_partition:
        exp_type += "__semisorted"
    else:
        exp_type += "__unsorted"

    dataset = datasets[dataset_name]

    dataset_folder = os.path.join(local_datasets_path, dataset.name)
    gdlc_folder = os.path.join(dataset_folder, exp_type)
    logs_folder = os.path.join("logs", dataset.name)
    os.makedirs(logs_folder, exist_ok=True)

    wandb_runs_path = os.path.join("logs", "wandb_runs")
    os.makedirs(wandb_runs_path, exist_ok=True)

    labels_mapping = {0: "Normal", 1: "Attack"}
    num_classes = 2
    if multi_class:
        with open(os.path.join(dataset_folder, "labels_names.pkl"), "rb") as f:
            labels_names = pickle.load(f)
        labels_mapping = labels_names[0]
    num_classes = len(labels_mapping)

    dataset_kwargs = dict(
        sequence_length=sequence_length,
        stride=stride,
        using_masking=False,
        masked_class=2,
        num_workers=0,
        device='cuda' if torch.cuda.is_available() else "cpu"
    )

    data_module = LitDataModule(
        gdlc_folder=gdlc_folder,
        dataset=dataset,
        batch_size=batch_size,
        multi_class=multi_class,
        network_features=network_features[dataset.centralities_set-1],
        **dataset_kwargs)
    data_module.setup()

    criterion = nn.CrossEntropyLoss(weight=data_module.class_weights)

    my_models = {
        "fnn": NIDSFNN(hidden_units=fnn_hidden_units, num_features=data_module.num_features, num_classes=num_classes, dropout=0.2, use_bn=True),
        "cnn": NIDSCNN(out_channels=cnn_out_channels_list, num_features=data_module.num_features, num_classes=num_classes, dropout=dropout),
        # "gru": NIDSGRU(num_features=data_module.num_features, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, num_classes=num_classes, dropout=dropout),
        # "lstm": NIDSLSTM(num_features=data_module.num_features, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, num_classes=num_classes, dropout=dropout),
        # "cnn_lstm": CNN_LSTM(out_channels=cnn_out_channels_list, seq_length=64, num_classes=num_classes, lstm_hidden_size=rnn_hidden_size, lstm_num_layers=rnn_num_layers, lstm_dropout=dropout, final_dropout=dropout),
    }

    config = {
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "fnn_hidden_units": fnn_hidden_units,
        "cnn_out_channels_list": cnn_out_channels_list,
        "rnn_num_layers": rnn_num_layers,
        "rnn_hidden_size": rnn_hidden_size,
        "activation": activation.__name__,
        "dropout": dropout,
        "multi_class": multi_class,
        "early_stopping_patience": early_stopping_patience,
        "sequence_length": sequence_length,
        "stride": stride,
    }

    for model_name, model in my_models.items():
        graph_model = LitClassifier(
            model=model,
            criterion=criterion,
            learning_rate=learning_rate,
            config=config,
            model_name=model_name,
            labels_mapping=labels_mapping,
            weight_decay=weight_decay,
            multi_class=True,
            label_col=dataset.label_col,
            class_num_col=dataset.class_num_col,
            batch_size=batch_size
        )

        data_module.set_model_type(model_type=model_name)

        wandb_logger = WandbLogger(
            project=f"GNN-Analysis-{dataset.name}",
            name=model_name,
            config=config,
            save_dir=wandb_runs_path
        )

        acc_checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            filename="best-val-acc-{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            verbose=False,
        )
        f1_checkpoint_callback = ModelCheckpoint(
            monitor="val_f1_score",
            mode="max",
            filename="best-val-f1-{epoch:02d}-{val_f1_score:.2f}",
            save_top_k=1,
            verbose=False,
        )
        loss_checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best-val-loss-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            verbose=False,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            verbose=False,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            num_sanity_val_steps=0,
            log_every_n_steps=0,
            callbacks=[
                acc_checkpoint_callback,
                f1_checkpoint_callback,
                loss_checkpoint_callback,
                early_stopping_callback
            ],
            default_root_dir=logs_folder,
            logger=wandb_logger if using_wandb else None,
        )

        trainer.fit(graph_model, datamodule=data_module)
        best_loss = loss_checkpoint_callback.best_model_path
        best_acc = acc_checkpoint_callback.best_model_path
        best_f1 = f1_checkpoint_callback.best_model_path

        graph_model.test_prefix = "best_loss"
        trainer.test(graph_model, datamodule=data_module, ckpt_path=best_loss)

        graph_model.test_prefix = "best_acc"
        trainer.test(graph_model, datamodule=data_module, ckpt_path=best_acc)

        graph_model.test_prefix = "best_f1"
        trainer.test(graph_model, datamodule=data_module, ckpt_path=best_f1)

        wandb.finish()


if __name__ == "__main__":
    main()
