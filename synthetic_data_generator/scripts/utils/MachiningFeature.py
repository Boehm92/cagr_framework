from synthetic_data_generator.scripts.geometric_primitives.ORing import ORing
from synthetic_data_generator.scripts.geometric_primitives.Hole import Hole
from synthetic_data_generator.scripts.geometric_primitives.Rectangle import Rectangle
from synthetic_data_generator.scripts.geometric_primitives.Triangle import Triangle
from synthetic_data_generator.scripts.geometric_primitives.SixSide import SixSide
from synthetic_data_generator.scripts.geometric_primitives.Round import Round
from synthetic_data_generator.scripts.geometric_primitives.Chamfer import Chamfer
from synthetic_data_generator.scripts.geometric_primitives.SlantedThroughStep import SlantedThroughStep
from synthetic_data_generator.scripts.geometric_primitives.CircularEnd import CircularEnd


class MachiningFeature:
    def __init__(self, machining_feature_id, limit):
        self.machining_feature_id = machining_feature_id
        self.limit = limit
        self.machining_feature = [ORing, Hole, Rectangle, Triangle, SixSide, Round, Chamfer, SlantedThroughStep,
                                  CircularEnd]

    def create(self):
        return self.machining_feature[self.machining_feature_id](self.limit).transformation()
