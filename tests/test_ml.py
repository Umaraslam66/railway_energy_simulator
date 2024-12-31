import pytest
import numpy as np
import pandas as pd
from railway_energy_simulator.src.ml.predictor import EnergyPredictor
from railway_energy_simulator.src.ml.optimizer import ScheduleOptimizer
from railway_energy_simulator.src.ml.patterns import PatternAnalyzer