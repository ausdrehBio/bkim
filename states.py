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
        self.store('epochs', 5)

        self.log("Reading data...")
        datafile = '/mnt/input/pneu.npz'
        train_loader, test_loader = get_dataloaders(datafile)   # raised error. Hier Pfad..
        self.store('data', train_loader)

        self.log('Initialising model...')
        # Fixed seed so all clients start with the same model
        torch.manual_seed(42)
        model = FederatedCNN(1, 1, device=torch.device('cpu'))
        self.store('model', model)

        return TRAIN_STATE


@app_state(TRAIN_STATE, role=Role.BOTH)
class TrainState(AppState):

    def register(self):
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)
        self.register_transition(TRAIN_STATE, role=Role.PARTICIPANT)

    def run(self):
        self.log("Waiting for aggregated parameters from coordinator...")
        model = self.load('model')

        # get initial parameters
        if self.load('iteration') == 0:
            # initially no aggregated parameters are available
            # but all clients start with the same model
            parameters = model.get_parameters()
        else:
            if self.is_coordinator:
                parameters = self.load('aggregated_parameters')
            else:    
                parameters = self.await_data()

        self.log('Initialising local model with new aggregated global parameters...')
        set_model_params(model, parameters)

        self.log('Training local model...')
        data = self.load('data')

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7)
        model.train_epoch('train', data, optimizer, criterion)
        self.log('Finished training local model...')

        local_parameters = model.get_parameters()
        self.log('Sending data to coordinator...')
        self.send_data_to_coordinator(local_parameters)  # gather data in Aggregate state

        epochs = self.load('epochs')
        iteration = self.load('iteration')
        
        if iteration <= epochs:
            if self.is_coordinator: 
                return AGGREGATE_STATE
            return TRAIN_STATE
        return TERMINAL_STATE


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
            self.broadcast_data(aggregated_parameters, send_to_self=False)
            return TRAIN_STATE  
        return WRITE_STATE


@app_state(WRITE_STATE, role=Role.COORDINATOR)
class WriteState(AppState):

        def register(self):
            self.register_transition(TERMINAL_STATE, role=Role.COORDINATOR)

        def run(self):
            self.log('Writing model to file...')
            model = self.load('model')
            torch.save(model.state_dict(), '/mnt/output/model.pt')
            return TERMINAL_STATE
