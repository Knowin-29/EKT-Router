from abc import ABC, abstractmethod


class BenchmarkEvaluator(ABC):
    def __init__(self, data_path, split="test"):
        self.data_path = data_path
        self.split = split

    @abstractmethod
    def evaluate(self, pipeline):
        pass
