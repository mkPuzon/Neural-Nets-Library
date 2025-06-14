'''run_model.py

Full pipline of loading in data, creating model, and running tests.

MKP -June 2025
'''
import tensorflow as tf

class Runner:
    
    def __init__(self, dataset, network) -> None:
        self.dataset = get_dataset(dataset)
        self.network = get_network(network)