from __future__ import annotations

from typing import Callable

from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.utils.typing import Config, PolicyID

MakeLearnerFunc = Callable[[Config, EnvInfo, Tensor, PolicyID, ParameterServer], Learner]


class LearnerFactory:
    def __init__(self):
        self.make_learner_func = Learner

    def register_learner_factory(self, make_learner_func):
        self.make_learner_func = make_learner_func
