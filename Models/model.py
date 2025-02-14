import torch  
import torchvision.models as models  
import torch.nn as nn
import timm  


class Efficientnet_b0_network(nn.Module):
    def __init__(self, model_name='efficientnet_b0', output_size=8):
        super(Efficientnet_b0_network, self).__init__()
        self.output_size = output_size  
        if model_name.lower() == 'efficientnet_b0':  
            self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=output_size)
 
    def forward(self, x):  
        return self.model(x)  
  
class Resnet_network(nn.Module):
    def __init__(self, model_name='resnet50', output_size=8):
        super(Resnet_network, self).__init__()
        self.output_size = output_size  
        if model_name.lower() == 'resnet18':  
            self.model = models.resnet18(pretrained=False)
        elif model_name.lower() == 'resnet50':  
            self.model = models.resnet50(pretrained=False)
        elif model_name.lower() == 'resnet101':  
            self.model = models.resnet101(pretrained=False)
        else:  
            raise ValueError(f"Unsupported model: {model_name}. Supported models are 'resnet18', 'resnet50', and 'resnet101'.")  
        num_ftrs = self.model.fc.in_features  
        self.model.fc = nn.Linear(num_ftrs, self.output_size)  
  
    def forward(self, x):  
        return self.model(x)  

class Large_network(nn.Module):  
    def __init__(self, model_name='vit_tiny_patch16_224', output_size=1000):  
        super(Large_network, self).__init__()
        self.output_size = output_size    
        self.Large_model = timm.create_model(model_name, pretrained=False)

        if "swin" in model_name:
            num_ftrs = self.Large_model.head.in_features  
            self.Large_model.head.fc = nn.Linear(num_ftrs, self.output_size) 
        else:
            num_ftrs = self.Large_model.head.in_features   
            self.Large_model.head = nn.Linear(num_ftrs, self.output_size)  
  
    def forward(self, x):  
        return self.Large_model(x)  


def vit_tiny_patch16(output_size):
    model = Large_network(model_name="vit_tiny_patch16_224",output_size=output_size)
    return model

def vit_small_patch16(output_size):
    model = Large_network(model_name="vit_small_patch16_224",output_size=output_size)
    return model

def vit_base_patch16(output_size):
    model = Large_network(model_name="vit_base_patch16_224",output_size=output_size)
    return model

def swin_tiny_patch16(output_size):
    model = Large_network(model_name="swin_tiny_patch4_window7_224",output_size=output_size)
    return model

def swin_small_patch16(output_size):
    model = Large_network(model_name="swin_small_patch4_window7_224",output_size=output_size)
    return model

def swin_base_patch16(output_size):
    model = Large_network(model_name="swin_base_patch4_window7_224",output_size=output_size)
    return model

def resnet18(output_size):
    model = Resnet_network(model_name="resnet18",output_size=output_size)
    return model

def resnet50(output_size):
    model = Resnet_network(model_name="resnet50",output_size=output_size)
    return model

def resnet101(output_size):
    model = Resnet_network(model_name="resnet101",output_size=output_size)
    return model

def Efficientnet_b0(output_size):
    model = Efficientnet_b0_network(model_name='efficientnet_b0',output_size=output_size)
    return model

vit_base_patch16_224 = vit_base_patch16
vit_small_patch16_224 = vit_small_patch16
vit_tiny_patch16_224 = vit_tiny_patch16

swin_small_patch4_window7_224 = swin_small_patch16
swin_base_patch4_window7_224 = swin_base_patch16
swin_tiny_patch4_window7_224 = swin_tiny_patch16

resnet101 = resnet101
resnet50 = resnet50
resnet18 = resnet18

Efficientnet_b0 = Efficientnet_b0

 # BST  "vit_base_patch16_224"   "vit_tiny_patch16_224"   "vit_small_patch16_224"
 # resnet18,resnet50
  
