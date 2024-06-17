from avocado import Avocado 
class avocado(object):
    def __init__(self):
        pass

    def load_model(self, chr="chr21"):
        self.model = Avocado.load(f"avocado-{chr}")

    

