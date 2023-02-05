from FeatureCloud.app.engine.app import AppState, app_state, Role
from model.model import FederatedCNN
from model.weights import get_avg_params, set_model_params
from model.dataset import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim


#TODO: SMPC = Secure Multi-Party Computation
#TODO: image-Dataset für Clients

INITIAL_STATE = 'initial'
TRAIN_STATE = 'train'
AGGREGATE_STATE = 'aggregate'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'
BROADCAST_STATE = 'broadcast'


@app_state(INITIAL_STATE)
class InitialState(AppState):
    """
    Initial state of the application. This state is responsible for
    """

    def register(self):
        self.register_transition(TRAIN_STATE, role=Role.BOTH)
        #self.register_transition(BROADCAST_STATE, role=Role.COORDINATOR)

    def run(self):

        self.store('iteration', 0)
        self.store('epochs', 1)

        self.log("Reading data...")
        datafile = '/mnt/input/pneu.npz'
        train_loader, val_loader, test_loader = get_dataloaders(datafile)   # raised error. Hier Pfad..

        self.log('Initialising model...')
        torch.manual_seed(42)  # Fixed seed so all clients start with the same model
        model = FederatedCNN(1, 1, device=torch.device('cpu'))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7)

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
        self.log("Waiting for aggregated parameters from coordinator...")
        model = self.load('model')
        stop = False

        # get initial parameters
        if self.load('iteration') == 0:
            # initially no aggregated parameters are available
            # but all clients start with the same model
            parameters = model.get_parameters()
        else:
            if self.is_coordinator:
                parameters = self.load('aggregated_parameters')
            else:    
                parameters, stop = self.await_data()

        self.log('Initialising local model with new aggregated global parameters...')
        set_model_params(model, parameters)

        if stop:
            self.store("model", model)
            return WRITE_STATE

        self.log('Training local model...')
        train_data = self.load('train_data')
        val_data = self.load('val_data')

        criterion = self.load('criterion')
        optimizer = self.load('optimizer')

        _ = model.train_epoch(train_data, optimizer, criterion)
        val_metrics = model.test_epoch(val_data, criterion)
        self.log('Finished training local model with test metrics: {}'.format(val_metrics))

        local_parameters = model.get_parameters()
        self.log('Sending data to coordinator...')
        self.send_data_to_coordinator(local_parameters)  # gather data in Aggregate state

        epochs = self.load('epochs')
        iteration = self.load('iteration')
        
        if iteration <= epochs:
            if self.is_coordinator: 
                return AGGREGATE_STATE
            return TRAIN_STATE
        return WRITE_STATE


@app_state(AGGREGATE_STATE, role=Role.COORDINATOR)
class AggregateState(AppState):

    def register(self):
        self.register_transition(TRAIN_STATE, role=Role.COORDINATOR)
        self.register_transition(WRITE_STATE, role=Role.COORDINATOR)

    def run(self):
        parameters = self.gather_data()
        aggregated_parameters = get_avg_params(parameters)
        self.store('aggregated_parameters', aggregated_parameters)

        epochs = self.load('epochs')
        current_iteration = self.load('iteration')
        current_iteration += 1
        self.store('iteration', current_iteration)

        if current_iteration <= epochs:
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
            # self.register_transition(TERMINAL_STATE, role=Role.PARTICIPANT)

        def run(self):
            model = self.load('model')

            self.log('Writing model to file...')
            test_data = self.load('test_data')
            criterion = self.load('criterion')

            test_metrics = model.test_epoch(test_data, criterion)

            # save test metrics to file
            with open('/mnt/output/test_metrics.txt', 'w') as f:
                f.write(str(test_metrics))

            self.log('Finished testing final model with test metrics: {}'.format(test_metrics))

            if self.is_coordinator:
                torch.save(model.state_dict(), '/mnt/output/model.pt')
                self.send_data_to_coordinator("DONE")
                self.gather_data()
            else:
                self.send_data_to_coordinator("DONE")
            return TERMINAL_STATE
