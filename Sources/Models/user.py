from .observableModel import ObservableModel

class User(ObservableModel):
    def __init__(self):
        super().__init__()
        
        self.user = None
    
    def set_user(self, user):
        self.user = user
        self.trigger_event("user_changed")
