import yaml
from FeatureCloud.app.engine.app import AppState, app_state, Role
from model.model import FederatedCNN
from model.weights import get_avg_params, set_model_params
from model.dataset import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim


INITIAL_STATE = 'initial'
TRAIN_STATE = 'train'
AGGREGATE_STATE = 'aggregate'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'
BROADCAST_STATE = 'broadcast'

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'


@app_state(INITIAL_STATE)
class InitialState(AppState):
    """
    Initial state of the application. This state is responsible for
    """

    def register(self):
        self.register_transition(TRAIN_STATE, role=Role.BOTH)

    def run(self):
        self.log("Reading config...")
        with open(f'{INPUT_DIR}/config.yml', 'r') as stream:
            config = yaml.safe_load(stream)

        self.log("Getting dataloaders...")
        datafile = config["data"]["file"]
        train_val_test_split = tuple(config["data"]["train_val_test_split"])
        batch_size = config["hyperparameter"]["batch_size"]
        split_seed = config["data"]["split_seed"]

        assert sum(train_val_test_split) == 1.0

        train_loader, val_loader, test_loader = get_dataloaders(
            path=f'{INPUT_DIR}/{datafile}',
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            seed=split_seed,
        )

        self.log('Initialising model...')
        model_seed = config["model"]["init_seed"]
        in_channels = config["model"]["in_channels"]
        out_channels = config["model"]["out_channels"]

        torch.manual_seed(model_seed)  # Fixed seed so all clients start with the same model
        model = FederatedCNN(in_channels, out_channels, device=torch.device('cpu'))

        self.log("Setting up hyperparameter...")
        learn_rate = config["hyperparameter"]["learn_rate"]
        momentum = config["hyperparameter"]["momentum"]
        epochs = config["hyperparameter"]["epochs"]
        positive_weight = config["hyperparameter"]["positive_weight"]

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)

        self.store('iteration', 1)
        self.store('epochs', epochs)
        self.store('model', model)
        self.store('criterion', criterion)
        self.store('optimizer', optimizer)
        self.store('train_data', train_loader)
        self.store('val_data', val_loader)
        self.store('test_data', test_loader)

        return TRAIN_STATE


@app_state(TRAIN_STATE, role=Role.BOTH)
class TrainState(AppState):

    def register(self):
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)
        self.register_transition(TRAIN_STATE, role=Role.PARTICIPANT)
        self.register_transition(WRITE_STATE, role=Role.PARTICIPANT)

    def run(self):
        iteration = self.load('iteration')
        self.log(f"Starting epoch {iteration}...")

        model = self.load('model')
        stop = False  # flag to indicate if training should stop

        # get initial parameters
        if self.load('iteration') == 1:
            # initially no aggregated parameters are available
            # but all clients start with the same model
            parameters = model.get_parameters()
        else:
            self.log("Waiting for aggregated parameters from coordinator...")
            if self.is_coordinator:
                parameters = self.load('aggregated_parameters')
            else:    
                parameters, stop = self.await_data()

        self.log(f"received stop flag: {stop}")
        self.log('Initialising local model with new aggregated global parameters...')
        set_model_params(model, parameters)

        if stop:
            self.log('Received stop flag from coordinator. Terminating...')
            self.store("model", model)
            return WRITE_STATE

        self.log('Training local model...')
        train_data = self.load('train_data')
        val_data = self.load('val_data')
        criterion = self.load('criterion')
        optimizer = self.load('optimizer')

        _, no_samples = model.train_epoch(train_data, optimizer, criterion, return_no_samples=True)
        val_metrics = model.test_epoch(val_data, criterion)
        self.log('Finished training local model with test metrics: {}'.format(val_metrics))

        # save test metrics to file
        with open(f'{OUTPUT_DIR}/val_metrics.txt', 'a+') as f:
            f.write(str(val_metrics) + "\n")

        local_parameters = model.get_parameters()
        self.log('Sending data to coordinator...')
        self.send_data_to_coordinator((local_parameters, no_samples))  # gather data in Aggregate state

        iteration += 1
        self.store('iteration', iteration)

        if self.is_coordinator:
            return AGGREGATE_STATE
        return TRAIN_STATE


@app_state(AGGREGATE_STATE, role=Role.COORDINATOR)
class AggregateState(AppState):

    def register(self):
        self.register_transition(TRAIN_STATE, role=Role.COORDINATOR)
        self.register_transition(WRITE_STATE, role=Role.COORDINATOR)

    def run(self):
        client_data = self.gather_data()
        parameters, no_samples = zip(*client_data)
        aggregated_parameters = get_avg_params(parameters, no_samples)
        self.store('aggregated_parameters', aggregated_parameters)

        epochs = self.load('epochs')
        iteration = self.load('iteration')

        if iteration <= epochs:
            stop = False
            self.broadcast_data((aggregated_parameters, stop), send_to_self=False)
            return TRAIN_STATE

        stop = True
        self.broadcast_data((aggregated_parameters, stop), send_to_self=False)
        return WRITE_STATE


@app_state(WRITE_STATE, role=Role.BOTH)
class WriteState(AppState):

    def register(self):
        self.register_transition(TERMINAL_STATE, role=Role.BOTH)

    def run(self):
        model = self.load('model')

        test_data = self.load('test_data')
        criterion = self.load('criterion')

        test_metrics = model.test_epoch(test_data, criterion)

        # save test metrics to file
        with open(f'{OUTPUT_DIR}/test_metrics.txt', 'w') as f:
            f.write(str(test_metrics))

        self.log('Finished testing final model with test metrics: {}'.format(test_metrics))

        self.send_data_to_coordinator("DONE")

        if self.is_coordinator:
            self.log('Writing model to file...')
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/model.pt')
            self.gather_data()  # wait for all participants to finish

        return TERMINAL_STATE
