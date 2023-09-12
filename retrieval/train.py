"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
import os
import torch
import utils
from torchvision import transforms
from pathlib import Path
import json

import config_file as conf
import model_builder
import data_setup
import engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a PyTorch image classification model using device-agnostic code.")
    parser.add_argument("--target_dir", default=conf.DST_DATA.as_posix(), help="directory containing all data")
    parser.add_argument("--train_file", default=conf.TRAIN_FILE.as_posix(),
                        help="file containing list of train samples")
    parser.add_argument("--test_file", default=conf.TEST_FILE.as_posix(), help="file containing list of test samples")
    parser.add_argument("--learning_rate", default=conf.LEARNING_RATE, help="learning rate used by the optimizer")
    parser.add_argument("--batch_size", default=conf.BATCH_SIZE, help="number of samples in a batch")
    parser.add_argument("--n_epochs", default=conf.NUM_EPOCHS, help="Number of epochs performed by the training")
    parser.add_argument("--n_hidden_units", default=conf.HIDDEN_UNITS, help="number of hidden units in the model")
    parser.add_argument("--model_name", default=conf.MODEL_NAME_DEFAULT, help="name of the finale .pth saved file")
    parser.add_argument("--device", default=conf.DEVICE, help="preferred device on which perform computation")

    args = parser.parse_args()

    target_dir = args.target_dir
    train_file = args.train_file
    test_file = args.test_file
    n_epochs = int(args.n_epochs)
    batch_size = int(args.batch_size)
    lr = float(args.learning_rate)
    hidden_units = int(args.n_hidden_units)
    model_name = args.model_name
    device = args.device

    # Create transforms
    data_transform = utils.transform_to_tensor

    # Create DataLoaders with help from data_setup.py
    print("\nCreate Dataloaders...")
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
        target_dir=target_dir,
        train_file=train_file, test_file=test_file,
        transform=data_transform,
        batch_size=batch_size,
        num_workers=conf.NUM_WORKERS
    )

    # Create model with help from model_builder.py
    print("\nCreate model...")
    model = model_builder.SimilarityNet(in_features=conf.IN_FEATURES, hidden_units=conf.HIDDEN_UNITS)
    # model.load_state_dict(torch.load(r"models/SimilarityNet_synth_data.pth"))
    model = model.to(device)

    # Set loss and optimizer

    # weighted cross-entropy
    # weights = torch.Tensor([0.01, 1]).to(device)
    # loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=conf.WEIGHT_DECAY)

    # Start training with help from engine.py
    print("\nStarting training...")
    results = engine.train_with_early_stopping(model=model,
                                               train_dataloader=train_dataloader,
                                               test_dataloader=test_dataloader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               epochs=n_epochs,
                                               device=device,
                                               patience=conf.PATIENCE,
                                               min_delta=conf.MIN_DELTA)

    # Save the model with help from utils.py
    print("\nSave model...")
    utils.save_model(model=model,
                     target_dir="models",
                     model_name=model_name)

    print("\nSave results data...")
    with open("results.txt", "w") as fp:
        json.dump(results, fp)

    print("\nPlotting loss curves...")
    utils.plot_loss_curves(results)
