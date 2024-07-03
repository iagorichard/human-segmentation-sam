from torchvision import models
#from SCNet import scnet
from torch import nn
import torch

class ModelGenerator():
    
    @staticmethod
    def get_dl_model(model_name, pretrained, device):

        assert model_name in ["resnet34", "vgg11", "alexnet",
                            "squeezenet", "densenet121"], "Model " + model_name + " is not defined yet!"
        
        if model_name == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 16)
        
        elif model_name == "vgg11":
            model = models.vgg11(pretrained=pretrained)
            model.classifier[6] = nn.Linear(4096, 16)
            
        elif model_name == "alexnet":
            model = models.alexnet(pretrained=pretrained)
            model.classifier[6] = nn.Linear(4096, 16)
            
        elif model_name == "squeezenet":
            model = models.squeezenet1_0(pretrained=pretrained)
            model.classifier[1] = nn.Conv2d(512, 16, kernel_size=(1,1), stride=(1,1))
            
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=pretrained)
            model.classifier = nn.Linear(1024, 16)
        
        model.to(device=device)
        
        return model