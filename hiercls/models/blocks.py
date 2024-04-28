from hiercls.utils.registry import registry
import torch.nn as nn
import ipdb

@registry.add_branch_net_to_registry('simple_branch')
class SimpleBranch(nn.Module):
    def __init__(self, input_feat_dim, num_classes):
        super().__init__()
        self.branch_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_feat_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
        )
        self.classification_head = nn.Sequential(nn.LeakyReLU(), nn.Linear(512, num_classes))

    def forward(self, x):
        branch_net_output = self.branch_net(x)
        logits = self.classification_head(branch_net_output)
        return logits
    

@registry.add_branch_net_to_registry('simple_branch_ff')
class SimpleBranchFF(SimpleBranch):
    def __init__(self, input_feat_dim, num_classes):
        super().__init__(input_feat_dim, num_classes)

    def forward(self, x, prev_branches_feat_sum=None):
        branch_net_output = self.branch_net(x)
        
        if prev_branches_feat_sum == None:
            ff = branch_net_output
        else:
            ff = branch_net_output + prev_branches_feat_sum
        
        logits = self.classification_head(ff)
        return (logits, ff)
    