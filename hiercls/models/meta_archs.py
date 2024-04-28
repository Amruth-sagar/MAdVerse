from hiercls.utils.registry import registry
import torch.nn as nn
from hiercls.models.backbones import *
from hiercls.models.blocks import *


#====================== LEAF-LEVEL-ONLY ======================#
    
@registry.add_meta_arch_to_registry('backbone_one_branch')
class CBackboneOneBranch(nn.Module):
    def __init__(self, model_config, hier_info):
        super().__init__()

        backbone = model_config.backbone.name
        branch = model_config.branch.name
        is_trainable = model_config.backbone.trainable
        
        self.num_leaves = len(hier_info.hierarchy_int)

        self.backbone = registry.mappings["backbones"][backbone](trainable=is_trainable)
        self.branch = registry.mappings["branch_nets"][branch]( self.backbone.out_feat_dim, self.num_leaves)

    def forward(self, x):
        cls_token = self.backbone(x)
        branch_outputs = []

        logits = self.branch(cls_token)

        branch_outputs.append(logits)

        return branch_outputs
    
    
#====================== MULTI-LEVEL ======================#

@registry.add_meta_arch_to_registry('backbone_branch')
class CBackboneCBranch(nn.Module):
    def __init__(self, model_config, hier_info):
        super().__init__()

        backbone = model_config.backbone.name
        branch = model_config.branch.name
        is_trainable = model_config.backbone.trainable

        self.num_levels = hier_info.num_levels
        self.num_cls_in_level = hier_info.num_cls_in_level

        self.backbone = registry.mappings["backbones"][backbone](trainable=is_trainable)
        self.branches = nn.ModuleList(
            [
                registry.mappings["branch_nets"][branch](
                    self.backbone.out_feat_dim, self.num_cls_in_level[i]
                )
                for i in range(self.num_levels)
            ]
        )

    def forward(self, x):
        cls_token = self.backbone(x)
        branch_outputs = []

        for i in range(self.num_levels):
            logits = self.branches[i](cls_token)

            branch_outputs.append(logits)

        return branch_outputs



# '_ff' means feature fusion, where cumulative sum of features from previous
# are used in current branch.


@registry.add_meta_arch_to_registry('backbone_branch_ff')
class CBackboneCBranchFF(nn.Module):
    def __init__(self, model_config, hier_info):
        super().__init__()

        backbone = model_config.backbone.name
        branch = model_config.branch.name
        is_trainable = model_config.backbone.trainable

        self.num_levels = hier_info.num_levels
        self.num_cls_in_level = hier_info.num_cls_in_level

        self.backbone = registry.mappings["backbones"][backbone](trainable=is_trainable)
        self.branches = nn.ModuleList(
            [
                registry.mappings["branch_nets"][branch](
                    self.backbone.out_feat_dim, self.num_cls_in_level[i]
                )
                for i in range(self.num_levels)
            ]
        )

    def forward(self, x):
        out_feats = self.backbone(x)
        branch_outputs = []

        logits, prev_branches_feat_sum = self.branches[0](out_feats)
        branch_outputs.append(logits)

        for i in range(1, self.num_levels):
            logits, prev_branches_feat_sum = self.branches[i](
                out_feats, prev_branches_feat_sum
            )

            branch_outputs.append(logits)
        return branch_outputs
