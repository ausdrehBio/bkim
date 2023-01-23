from FeatureCloud.app.engine.app import AppState, app_state, Role
from model.model import CNN
from model.weights import average_weights
from model.dataset import get_dataloaders

# FRAGE: wo wird wann was importiert? welche dateien in welchem zustand? jp mb


INITIAL_STATE = 'initial'
TRAIN_STATE = 'train'
AGGREGATE_STATE = 'aggregate'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'
BROADCAST_STATE = 'broadcast'


''' 
wir starten im INITIAL_STATE und schauen ob wir coordinator sind oder nicht 
wenn wir coordinator sind, dann gehen wir in den BROADCAST_STATE
wenn wir nicht coordinator sind, dann gehen wir in den TRAIN_STATE
'''
@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(TRAIN_STATE, role=Role.Participant)
        self.register_transition(BROADCAST_STATE, role=Role.Coordinator)

    def run(self):

        self.store('iteration', 0)
        self.store('epochs', 20)

        self.log("Reading data...")
        data = get_dataloaders()
        self.store('data', data)

        self.log('Initialising model...')
        model = CNN()
        self.store('model', model)

        if self.is_coordinator:
            return BROADCAST_STATE 
        return TRAIN_STATE


'''
wir sind coordinator und wollen die parameter verteilen
danach gehen wir in den TRAIN_STATE
'''
@app_state(BROADCAST_STATE, role=Role.COORDINATOR)
class BroadcastState(AppState):

    def register(self):
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)

    def run(self):
        model = self.load('model')
        parameters = model.get_parameters()
        self.broadcast(parameters, send_to_self=False) # damit können wir die parameter an alle Clients verteilen
        return AGGREGATE_STATE


'''
wir sind participant und wollen die parameter empfangen
wir nutzen die parameter um zu trainieren
die trained_parameter schicken wir an den coordinator
wir gehen über zum AGGREGATE_STATE
'''
@app_state(TRAIN_STATE, role=Role.Participant)
class TrainState(AppState):

    def register(self):
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)
        self.register_transition(WRITE_STATE)

    def run(self):
        self.log("Waiting for aggregated parameters from coordinator...")
        parameters = self.await_data()

        model = self.load('model')
        self.log('Initialising local model with new aggregated global parameters...')
        model.set_parameters(parameters)

        data = self.load('data')
        self.log('Training local model...')
        model.train(data)
        self.log('Finished training local model...')

        local_parameters = model.get_parameters
        self.log('Sending data to coordinator...')
        self.send_data_to_coordinator([local_parameters])

        epochs = self.load('epochs')
        iteration = self.load('iteration')
        
        if iteration <= epochs:
            if self.is_coordinator:
                return AGGREGATE_STATE
            return TRAIN_STATE
        return TERMINAL_STATE


'''
wir sind coordinator und wollen die trained_parameter empfangen
wir nutzen die trained_parameter um zu aggregieren
wenn wir noch nicht alle communicationrounds durch haben, dann verteilen wir die aggregated_parameter an alle Clients
wenn wir alle communicationrounds durch haben, dann gehen wir in den TERMINAL_STATE
'''
@app_state(AGGREGATE_STATE, role=Role.COORDINATOR)
class AggregateState(AppState):

    def register(self):
        self.register_transition(BROADCAST_STATE, role=Role.COORDINATOR)
        self.register_transition(TERMINAL_STATE, role=Role.COORDINATOR)

    def run(self):
        parameters = self.aggregate_data()
        model = self.load('model')
        aggregated_parameters = average_weights(parameters, model) #model NN wie TRAIN STATE jo, mb

        epochs = self.load('epochs')
        current_iteration = self.load('iteration')
        current_iteration += 1
        self.store('iteration', current_iteration)

        if (current_iteration <= epochs):
            self.broadcast(aggregated_parameters, send_to_self=False)
            return TRAIN_STATE
        return TERMINAL_STATE
