import os
import numpy as np
import madcad as mdc
from synthetic_data_generator.scripts.utils.MachiningFeature import MachiningFeature
from synthetic_data_generator.scripts.utils.CsgOperation import CsgOperation
from synthetic_data_generator.scripts.geometric_primitives.base_primitive import Cube
from synthetic_data_generator.scripts.utils.MachiningFeatureLabels import MachiningFeatureLabels


class DataGenerator:
    def __init__(self, config):
        self.cad_data_generation_start_cycle = config.cad_data_generation_start_cycle
        self.cad_data_generation_end_cycles = config.cad_data_generation_end_cycles
        self.max_machining_feature_count = config.max_machining_feature_count
        self.max_machining_feature_dimension = config.max_machining_feature_dimension
        self.max_base_primitive_dimension = config.max_base_primitive_dimension
        self.target_directory = config.target_directory
        self.select_machining_feature_id_random = config.select_machining_feature_id_random
        self.manual_selected_machining_feature_id = config.manual_selected_machining_feature_id

    def generate(self):
        for _model_id in range(self.cad_data_generation_start_cycle, self.cad_data_generation_end_cycles):
            _machining_feature_id_list = []
            _machining_feature_list = []
            _manufacturing_time = 0
            _new_cad_model = Cube(10, mdc.vec3(5, 5, 5)).transform()
            _machining_feature_count = np.random.randint(1, (self.max_machining_feature_count + 1))

            try:
                for _ in range(_machining_feature_count):
                    if self.select_machining_feature_id_random:
                        _machining_feature_id = np.random.randint(0, 9)
                    else:
                        _machining_feature_id = self.manual_selected_machining_feature_id

                    _machining_feature, _machining_feature_manufacturing_time = \
                        MachiningFeature(_machining_feature_id, self.max_machining_feature_dimension).create()

                    if _machining_feature_manufacturing_time <= 0:
                        raise ValueError("Manufacturing time is zero or below.")

                    _manufacturing_time += _machining_feature_manufacturing_time
                    _new_cad_model = CsgOperation(_new_cad_model, _machining_feature).difference()
                    _machining_feature_id_list.append(_machining_feature_id)
                    #     _machining_feature_list.append(_machining_feature)

                print("")
                print(f"Created CAD model {_model_id} with {_machining_feature_count} machining feature")
                print(f"machining feature: {_machining_feature_id_list}")
                print(f"manufacturing time: {_manufacturing_time}")
                mdc.write(_new_cad_model, os.getenv(self.target_directory) + "/" + str(_model_id) + ".stl")
                MachiningFeatureLabels(_machining_feature_list, _model_id, self.target_directory,
                                       _machining_feature_id_list, _manufacturing_time).write_manufacturing_time_file()
            except:
                # We use here a broad exception clause to avoid applying machining feature if not enough surface is
                # available
                print("")
                print(f"One or more machining feature for the CAD model {_model_id} were not feasible."
                      f" For CAD model {_model_id}, {_} from {_machining_feature_count} have been applied."
                      f" This can happen when not enough surface is available for the CSG difference operation."
                      f" Or, if the machining feature time variable is zero (Current machining feature time:"
                      f" {_machining_feature_manufacturing_time}), then an exception is thrown as well and the CAD "
                      f"model isn't saved. This can happen, if the random created machining feature volume is so small,"
                      f" that the manufacturing time after the round function has the value zero")

            del _new_cad_model
            del _machining_feature_id_list

