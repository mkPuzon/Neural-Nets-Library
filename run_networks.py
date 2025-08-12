'''run_networks.py

MKP - August 2025
'''
from datasets import load_dataset
from utils import get_model_type, get_data_params, get_net_params, save_model_wts

def run_experiments(networks_list, data_set_name, save_wts=False, verbose=False):
    data_params = get_data_params(data_set_name)
    data = load_dataset(data_set_name, data_params, verbose=verbose)
    
    hists = []
    for model in networks_list:
        if verbose:
            print(f"==== Running {model} on {data_set_name} ====")
        model_type = get_model_type(model)
        if model_type == 'cnn':
            from base_networks import CNN
            model = CNN(params=get_net_params(model))
        elif model_type == 'mlp':
            from base_networks import MLP
            model = MLP(params=get_net_params(model))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.build_model(data_params=data_params, print_summary=False)
        model.compile_model()
        
        hist = model.fit_model(x_train=data[0], y_train=data[1], 
                               x_val=data[2], y_val=data[3], 
                               save_wts=save_wts, verbose=verbose)
        
        hists.append(hist)
        
    return hists
        
        