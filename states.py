from FeatureCloud.app.engine.app import AppState, app_state, Role

INITIAL_STATE = 'initial'
TRAIN_STATE = 'train'
AGGREGATE_STATE = 'aggregate'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(TRAIN_STATE)

    def run(self):
        return TRAIN_STATE


# This state indicates that the client is currently training a model locally
@app_state(TRAIN_STATE)
class TrainState(AppState):

    def register(self):
        self.register_transition(AGGREGATE_STATE, role=Role.Coordinator)
        self.register_transition(WRITE_STATE)

    def run(self):
        # NN training code goes here
        if self.is_coordinator:
            return AGGREGATE_STATE
        return WRITE_STATE


# This state indicates that the coordinator is currently aggregating the results of all clients
@app_state(AGGREGATE_STATE)
class AggregateState(AppState):

    def register(self):
        self.register_transition(WRITE_STATE, role=Role.Coordinator)

    def run(self):
        # Aggregation code goes here
        return ''


# This state indicates that the client is currently sending its results after finishing local training
@app_state(WRITE_STATE)
class WriteState(AppState):

    def register(self):
        self.register_transition(TRAIN_STATE, role=Role.Participant)

    def run(self):
        return ''
