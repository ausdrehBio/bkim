from FeatureCloud.app.engine.app import AppState, app_state, Role
from model.model import FederatedCNN
from model.weights import average_weights
from model.dataset import get_dataloaders
import os





#TODO: SMPC = Secure Multi-Party Computation
#TODO: image-Dataset für Clients

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
#datapath = r'jonaspfiffner/featurecloud_data/pneumoniamnist.npz' #path to featurecloud data folder
@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(TRAIN_STATE, role=Role.BOTH)
        #self.register_transition(BROADCAST_STATE, role=Role.COORDINATOR)

    def run(self):

        self.store('iteration', 0)
        self.store('epochs', 20)

        self.log("Reading data...")
        
        #########
        # Datapath /mnt/input/*.npz file 
        
        datafile = '/mnt/input/pneu.npz'
        data = get_dataloaders(datafile)   # raised error. Hier Pfad..
        self.store('data', data)

        self.log('Initialising model...')
        model = FederatedCNN(1,1)
        self.store('model', model)

        #if self.is_coordinator:
        #    return BROADCAST_STATE 
        return TRAIN_STATE



'''
wir sind coordinator und wollen die parameter verteilen
danach gehen wir in als coordinator in den AGGREGATE_STATE
'''
'''
@app_state(BROADCAST_STATE, role=Role.BOTH)
class BroadcastState(AppState):

    def register(self):
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)

    def run(self):
        model = self.load('model')
        parameters = model.get_parameters()
        self.broadcast(parameters, send_to_self=False) # damit können wir die parameter an alle Clients verteilen
        return AGGREGATE_STATE
'''

'''
wir sind participant und wollen die parameter empfangen
wir nutzen die parameter um zu trainieren
die trained_parameter schicken wir an den coordinator
wir gehen über zum AGGREGATE_STATE
'''
@app_state(TRAIN_STATE, role=Role.BOTH)
class TrainState(AppState):

    def register(self):
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR) # können wir erneut in den train_state gehen?
        self.register_transition(TRAIN_STATE, role=Role.PARTICIPANT)

    def run(self):
        self.log("Waiting for aggregated parameters from coordinator...")
        if self.is_coordinator:
            #send data to participants
            parameters = self.load('aggregated_parameters')
            self.broadcast(parameters, send_to_self=False)
        else:    
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
        self.send_data_to_coordinator([local_parameters]) #gather data in Aggregate state

        epochs = self.load('epochs')
        iteration = self.load('iteration')
        
        if iteration <= epochs:
            if self.is_coordinator: 
                return AGGREGATE_STATE
            return TRAIN_STATE
        return TERMINAL_STATE
        '''
        wäre es möglich, dass der Participant immer nur von TrainState zu TrainState geht?
        und wir die Entscheidung zwischen weiter trainieren und terminieren auschließlich im Coordinator machen?
        --> "If the coordinator transitions into the 'terminal' state, the whole computation will be shut down."
        '''


'''
wir sind coordinator und wollen die trained_parameter empfangen
wir nutzen die trained_parameter um zu aggregieren
wenn wir noch nicht alle communicationrounds durch haben, dann verteilen wir die aggregated_parameter an alle Clients
wenn wir alle communicationrounds durch haben, dann gehen wir in den TERMINAL_STATE
'''
@app_state(AGGREGATE_STATE, role=Role.COORDINATOR)
class AggregateState(AppState):

    def register(self):
        self.register_transition(TRAIN_STATE, role=Role.COORDINATOR)
        self.register_transition(TERMINAL_STATE, role=Role.COORDINATOR)

    def run(self):
        parameters = self.gather_data()
        model = self.load('model')
        aggregated_parameters = average_weights(parameters, model) 

        epochs = self.load('epochs')
        current_iteration = self.load('iteration')
        current_iteration += 1
        self.store('iteration', current_iteration)

        if (current_iteration <= epochs):
            self.broadcast(aggregated_parameters, send_to_self=False)
            return TRAIN_STATE  
        return TERMINAL_STATE
