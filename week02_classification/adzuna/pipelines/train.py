import torch
from adzuna.config import *
from adzuna.pipelines.preprocess import BatchPreprocessor
from adzuna.pipelines.early_stopping import EarlyStopper
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR


class TrainExecutor:
    def __init__(self, model, criterion, optimizer, batch_preprocessor: BatchPreprocessor):
        self.batch_preprocessor = batch_preprocessor
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)
        self.early_stopper = EarlyStopper(patience=3, min_delta=0.01)

    def val_epoch(self, data, batch_size=BATCH_SIZE, name="", **kw):
        squared_error = abs_error = num_samples = 0.0
        self.model.eval()
        running_loss = 0.0
        num_of_iterations = 0
        with torch.no_grad():
            for batch in self.batch_preprocessor.iterate_minibatches(data, batch_size=batch_size, shuffle=False, **kw):
                batch_pred = self.model(batch)
                squared_error += torch.sum((batch_pred - batch[TARGET_COLUMN]).pow(2))
                abs_error += torch.sum(torch.abs(batch_pred - batch[TARGET_COLUMN]))
                num_samples += len(batch_pred)

                loss = self.criterion(batch_pred, batch[TARGET_COLUMN])
                running_loss += loss.item()
                num_of_iterations += 1

        mse = squared_error.detach().cpu().numpy() / num_samples
        mae = abs_error.detach().cpu().numpy() / num_samples
        val_loss = running_loss / num_of_iterations
        print("%s results:" % (name or ""))
        print("Mean square error: %.5f" % mse)
        print("Mean absolute error: %.5f" % mae)
        return mse, mae, val_loss

    def train_epoch(self, data_train):
        self.model.train()
        num_of_iterations = 0
        running_loss = 0.0
        squared_error = abs_error = num_samples = 0.0
        for i, batch in tqdm(enumerate(self.batch_preprocessor.iterate_minibatches(data_train, batch_size=BATCH_SIZE)),
                             total=len(data_train) // BATCH_SIZE):
            pred = self.model(batch)
            loss = self.criterion(pred, batch[TARGET_COLUMN])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            num_of_iterations += 1
            squared_error += torch.sum((pred - batch[TARGET_COLUMN]).pow(2))
            abs_error += torch.sum(torch.abs(pred - batch[TARGET_COLUMN]))
            num_samples += len(pred)

            with torch.no_grad():
                running_loss += loss.item()
        train_loss = running_loss / num_of_iterations
        train_mse = squared_error.detach().cpu().numpy() / num_samples
        train_mae = abs_error.detach().cpu().numpy() / num_samples

        return train_mse, train_mae, train_loss

    def train(self, num_of_epochs, data_train, data_val):
        metrics_data = []
        for epoch in range(num_of_epochs):
            print(f"EPOCH: {epoch}")
            train_mse, train_mae, train_loss = self.train_epoch(data_train)
            self.scheduler.step()
            val_mse, val_mae, val_loss = self.val_epoch(data_val)

            if self.early_stopper.early_stop(val_mae):
                print('Early stopping at epoch: ', epoch, ' with val_mae: ', val_mae, ' and train_mae: ', train_mae, '')
                break

            metrics_data.append(
                {
                    'train_mse': train_mse,
                    'train_mae': train_mae,
                    'train_loss': train_loss,
                    'val_mse': val_mse,
                    'val_mae': val_mae,
                    'val_loss': val_loss
                }
            )
        return metrics_data
