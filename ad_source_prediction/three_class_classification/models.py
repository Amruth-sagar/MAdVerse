import torch.nn as nn
from transformers import ViTModel, ConvNextModel
from lavis.models import load_model_and_preprocess

model_name_to_class = {}

def register_model_name(name):
    def decorater(cls):
        model_name_to_class[name] = cls
        return cls
    return decorater

class ConvClsHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cls_head = nn.Sequential(
             nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2),
             nn.MaxPool2d(kernel_size=3),
             nn.LeakyReLU(),
             nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, stride=(2,3)),
             nn.MaxPool2d(kernel_size=(3,4)),
             nn.LeakyReLU(),
             nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4),
             nn.MaxPool2d(kernel_size=(1,3)),
             nn.LeakyReLU(),
             nn.Flatten(),
             nn.Linear(16,num_classes)
             )
    def forward(self, x):
        return self.cls_head(x)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_prob):
        super().__init__()

        # Dropout layers after activation.
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 150),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(150, 50),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(50,num_classes)
        )
    def forward(self, x):
        return self.cls_head(x)

@register_model_name('vit_with_cls_head')
class ViTClsHead(nn.Module):
    def __init__(self, pretrained, feature_dim, num_classes, dropout_prob, is_trainable=False):
        super().__init__()
        self.feat_ext = ViTModel.from_pretrained(pretrained)
        if not is_trainable:
            for param in self.feat_ext.parameters():
                param.requires_grad = False
        self.cls_head = ClassificationHead(feature_dim, num_classes, dropout_prob)

    def forward(self, x):
        cls_token = self.feat_ext(x).pooler_output
        return self.cls_head(cls_token)

@register_model_name('convnext_with_cls_head')
class ConvNeXtClsHead(nn.Module):
    def __init__(self, pretrained, feature_dim, num_classes, dropout_prob, is_trainable=False):
        super().__init__()
        self.feat_ext = ConvNextModel.from_pretrained(pretrained)
        if not is_trainable:
            for param in self.feat_ext.parameters():
                param.requires_grad = False
        self.cls_head = ClassificationHead(feature_dim, num_classes, dropout_prob)
        
    def forward(self, x):
        cls_token = self.feat_ext(x).pooler_output
        return self.cls_head(cls_token)
    
@register_model_name('blipv2_with_cls_head')
class ConvNeXtClsHead(nn.Module):
    def __init__(self, pretrained, feature_dim, num_classes, dropout_prob, is_trainable=False):
        super().__init__()
        self.cls_head = ClassificationHead(feature_dim, num_classes, dropout_prob)
        self.model, _, _ = load_model_and_preprocess(
            name=pretrained, model_type="pretrain", is_eval=eval, use_auth_token=True
        )
        if not is_trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # the output obtained by the model is image features of size(batch size,32,768) 
        # we are using conv1d to obtain pooled outputs
        self.conv1d = nn.Conv1d(32, 1, 3, 1, 1)

    def forward(self, x):
        model_input = {"image": x}

        features_images = self.model.extract_features(model_input, mode="image")
        pooled_features = self.conv1d(features_images.image_embeds)

        # From [batch, 1, 768] to [batch, 768]
        pooled_features.squeeze_(1)
        cls_token = self.feat_ext(x).pooler_output
        return self.cls_head(cls_token)