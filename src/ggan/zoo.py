'''Defines the ModelZoo class, which allows for easy switching of models'''

from .blgan import BetterSharedFFBilinearGenerator, BaselineGenerator
from .svd import ModdedSharedSvdGenerator
from .ablation import FCDiscriminator
from .residual import FixedResidualGcnDiscriminator, RepairedResidualGcnDiscriminator

class ModelZoo:
    def __init__(self):
        self.models = {
            'BetterSharedFfBilinearGenerator': BetterSharedFFBilinearGenerator,
            'FixedResidualGcnDiscriminator': FixedResidualGcnDiscriminator,
            'RepairedResidualGcnDiscriminator': RepairedResidualGcnDiscriminator,
            'BaselineGenerator': BaselineGenerator,
            'ModdedSharedSvdGenerator': ModdedSharedSvdGenerator,
            'FCDiscriminator': FCDiscriminator
        }

    def get_model(self, model_name):
        '''Given a model name, returns the model class'''
        if model_name not in self.models:
            raise Exception(
                f'Unknown model specified. Valid options are: {self.models.keys()}')
        return self.models[model_name]

    def has_model(self, model_name):
        '''Given a model name, return whether or not it exists'''
        return model_name in self.models

