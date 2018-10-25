import pydart2 as pydart
import numpy as np
import creature_controllers
from sklearn.preprocessing import normalize
from pydart2.gui import trackball

class BaseSimulator(pydart.World):
    def __init__(self, inputVector):
        super(BaseSimulator, self).__init__(0.0003, './data/skel/longWorm.skel')

        self.controller = creature_controllers.Controller(self.skeletons[1], inputVector)
        self.skeletons[1].set_controller(self.controller)

