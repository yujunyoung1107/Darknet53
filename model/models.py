from model.Darknet53 import Darknet53

def get_model(model_name):
    if(model_name == "Darknet53"):
        return Darknet53
    else:
        print('unknown')