from hiercls.utils.registry import registry
import torch.nn as nn
from transformers import ViTModel, ConvNextModel
from lavis.models import load_model_and_preprocess
from torch import float32
import torch
import ipdb
import torchvision.models as models

@registry.add_backbone_to_registry('vit-base-patch16-224')
class ViTbase(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.pretrained_ckpt_path = "google/vit-base-patch16-224"
        self.feat_ext = ViTModel.from_pretrained(self.pretrained_ckpt_path)
        self.out_feat_dim = 768

        if not trainable:
            for param in self.feat_ext.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # The output shape for this particular model is [batch, 768]
        # The pooler output is just CLS token from a linear layer and tanh activation
        return self.feat_ext(x).pooler_output

@registry.add_backbone_to_registry('vit-large-patch16-224-in21k')
class ViTLarge(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.pretrained_ckpt_path = "google/vit-large-patch16-224-in21k"
        self.feat_ext = ViTModel.from_pretrained(self.pretrained_ckpt_path)
        self.out_feat_dim = 1024

        if not trainable:
            for param in self.feat_ext.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # The output shape for this particular model is [batch, 1024]
        # The pooler output is just CLS token from a linear layer and tanh activation
        return self.feat_ext(x).pooler_output



@registry.add_backbone_to_registry('convnext-base-224-22k')
class ConvNextbase(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.pretrained_ckpt_path = "facebook/convnext-base-224-22k"
        self.feat_ext = ConvNextModel.from_pretrained(self.pretrained_ckpt_path)
        self.out_feat_dim = 1024

        if not trainable:
            for param in self.feat_ext.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # The output shape for this particular model is [batch, 1024]
        return self.feat_ext(x).pooler_output
    

@registry.add_backbone_to_registry('convnext-large-224-22k-1k')
class ConvNextLarge(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.pretrained_ckpt_path = "facebook/convnext-large-224-22k-1k"
        self.feat_ext = ConvNextModel.from_pretrained(self.pretrained_ckpt_path)
        self.out_feat_dim = 1536

        if not trainable:
            for param in self.feat_ext.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # The output shape for this particular model is [batch, 1536]
        return self.feat_ext(x).pooler_output
    
@registry.add_backbone_to_registry('blip-v2')
class BLIPv2(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()

        self.model, _, _ = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain", is_eval=eval
        )
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # the output obtained by the model is image features of size(batch size,32,768) 
        # we are using conv1d to obtain pooled outputs
        self.conv1d = nn.Conv1d(32, 1, 3, 1, 1)

        self.out_feat_dim = 768

    def forward(self, x):
        model_input = {"image": x}

        features_images = self.model.extract_features(model_input, mode="image")
        pooled_features = self.conv1d(features_images.image_embeds)

        # From [batch, 1, 768] to [batch, 768]
        pooled_features.squeeze_(1)
        return pooled_features

        
