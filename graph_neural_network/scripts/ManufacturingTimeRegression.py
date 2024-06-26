import os
import time
import torch
import wandb
import optuna
from torch_geometric.loader import DataLoader
from graph_neural_network.scripts.utils.HyperParameter import HyperParameter


class ManufacturingTimeRegression:
    def __init__(self, config, trial):
        self.max_epoch = config.max_epoch
        self.training_dataset = config.training_dataset
        self.test_dataset = config.test_dataset
        self.network_model = config.network_model
        self.network_model_id = config.network_model_id
        self.amount_training_data = config.amount_training_data
        self.amount_validation_data = config.amount_validation_data
        self.trial = trial
        self.hyper_parameters = HyperParameter(trial, config.network_model_id)
        self.device = config.device
        self.project_name = config.project_name
        self.study_name = config.study_name

    def training(self):
        _best_accuracy = 0

        self.training_dataset.shuffle()
        _train_loader = DataLoader(self.training_dataset[:self.amount_training_data],
                                   batch_size=self.hyper_parameters.batch_size, shuffle=True, drop_last=True)
        _val_loader = DataLoader(self.training_dataset[self.amount_training_data:
                                                       self.amount_training_data + self.amount_validation_data],
                                 batch_size=self.hyper_parameters.batch_size, shuffle=True, drop_last=True)

        _network_model = self.network_model(self.training_dataset, self.device, self.hyper_parameters).to(self.device)
        print(_network_model)
        print(self.device)

        # Configuring learning functions
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(_network_model.parameters(), lr=self.hyper_parameters.learning_rate)

        # Setting up hyperparameter function and wandb
        _config = dict(self.trial.params)
        _config["trial.number"] = self.trial.number
        wandb.init(project=self.project_name, entity="boehm92", config=_config, group=self.study_name, reinit=True)

        # Training
        for epoch in range(1, self.max_epoch):
            loss, avg_mae_loss = _network_model.train_loss(_train_loader, criterion, optimizer)
            training_loss, avg_mae_training_loss, avg_rmse_train_loss, avg_r2_train_loss =\
                _network_model.val_loss(_train_loader, criterion)
            val_loss, avg_mae_val_loss, avg_rmse_val_loss, avg_r2_val_loss = \
                _network_model.val_loss(_val_loader, criterion)

            wandb.log({'loss': loss, 'training_loss': training_loss, 'val_los': val_loss,
                       'avg_mae_loss': avg_mae_loss, 'avg_mae_training_loss': avg_mae_training_loss,
                       'avg_mae_val_loss': avg_mae_val_loss})

            # if (_best_accuracy < val_f1) & ((val_loss - training_loss) < 0.04):
            #     torch.save(_network_model.state_dict(), os.getenv('WEIGHTS') + '/weights.pt')
            #     _best_accuracy = val_f1
            #     print("Saved model due to better found accuracy")

            if self.trial.should_prune():
                wandb.run.summary["state"] = "pruned"
                wandb.finish(quiet=True)
                raise optuna.exceptions.TrialPruned()

            print(f'Epoch: {epoch:03d}, loss: {loss:.4f}, training_loss: {training_loss:.4f}, val_los: {val_loss:.4f},'
                  f'avg_mae_loss: {avg_mae_loss:.4f}, avg_mae_training_loss: {avg_mae_training_loss:.4f}, '
                  f'avg_mae_val_loss: {avg_mae_val_loss:.4f},'
                  f' avg_rmse_train_loss: {avg_rmse_train_loss:.4f}, avg_rmse_train_loss: {avg_rmse_val_loss:.4f},'
                  f', avg_r2_train_loss: {avg_r2_train_loss:.4f}, avg_r2_val_loss: {avg_r2_val_loss:.4f}')

        wandb.run.summary["Final F-Score"] = val_loss
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)

        return val_loss

    def test(self):
        _test_loader = DataLoader(self.test_dataset, batch_size=self.hyper_parameters.batch_size, shuffle=False,
                                  drop_last=True)
        _network_model = self.network_model(self.test_dataset, self.device, self.hyper_parameters).to(self.device)
        _network_model.load_state_dict(torch.load(os.getenv('WEIGHTS') + '/weights.pt'),)
        print("Graph neural network: ", _network_model)
        start_time = time.time()
        _test_f1, flops, params = _network_model.accuracy(_test_loader)
        end_time = time.time()

        total_time = end_time - start_time

        print("F1-Score: ", _test_f1)
        print("Runtime: ", total_time)
        print("Average FLOPS: ", flops)
        print("Average Parameters: ", params)
