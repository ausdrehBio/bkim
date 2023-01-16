from FeatureCloud.app.engine.app import AppState, app_state, Role

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
        randomly_initialized_parameter = "lol"
        self.broadcast(randomly_initialized_parameter, send_to_self=False) # damit können wir die parameter an alle Clients verteilen
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
        parameter = self.await_data() # wir warten auf die parameter vom coordinator
        # NN training code goes here using parameter
        self.send_data_to_coordinator(trained_parameter) # wir schicken die trained_parameter an den coordinator
        if self.is_coordinator:
            return AGGREGATE_STATE
        return WRITE_STATE



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
        parameter = self.aggregate_data()
        # Aggregation code goes here
        if(communicationrounds != 0):
            self.broadcast(weighted_parameter, send_to_self=False)
            return TRAIN_STATE
        return TERMINAL_STATE

