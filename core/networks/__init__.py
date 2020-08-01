import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_flow_paper import Model_flow

def get_model(mode):
    if mode == 'flow':
        return Model_flow
    else:
        raise ValueError('Mode {} not found.'.format(mode))
