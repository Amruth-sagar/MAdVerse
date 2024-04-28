from datasets import *
from dataloader import *
import torch.nn as nn
from models import *
from metrics import *
import wandb
from utils import *
from optim_lrsched import *
from pprint import pprint
from train_utils import *
from dataset import *
from transformers import ViTFeatureExtractor, ConvNextFeatureExtractor
import torch
# Loading the configuration
config = load_config('/home2/rishabh.s/ADS_NON_ADS_Classifier/classification/config.yaml')

# Seeding all the important random packages.
seed_everything(42)

def generate_run_name(config):
    """
        Generates the name of the wandb run using config.
    """
    return 'feat.ext.{}.lr.{}.CLS.lr.{}.dp.{}.T_0.{}.eta_min.{}.b.{}.e{}'.format(
        config['model']['feat_extractor'],
        config['lr']['feat_ext_lr'],
        config['lr']['base_lr'],
        config['arch'][config['model']['feat_extractor']]['args']['dropout_prob'],
        config['lr']['T_0'],
        config['lr']['eta_min'],
        config['model']['train_bs'],
        config['model']['epochs']
    )

def merge_configs(sweep_config, config):
    """
        Merging the hyperparameters from wandb config
    """
    config['lr']['T_0'] = sweep_config['T_0']
    config['lr']['feat_ext_lr'] = sweep_config['feat_ext_lr']
    config['lr']['base_lr'] = sweep_config['base_lr']

    feat_ext_config = config['arch'][config['model']['feat_extractor']]
    feat_ext_config['args']['dropout_prob'] = sweep_config['dropout_prob']


def sweep_agent_manager():
    wandb.init()
    sweep_config = wandb.config

    print("\n ### SWEEP CONFIG ###\n")
    pprint(sweep_config)
    print("\n####################\n\n")

    merge_configs(sweep_config, config)
    run_name = generate_run_name(config)
    wandb.run.name = run_name
    trainer(config, run_name, no_sweep = False)



def trainer(config, run_name=config['wandb']['run_name'], no_sweep = True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_log = config['wandb']['wandb_log']

    # If sweep is going on, no_sweep is false, so no need to create run again
    # if wandb_log and no_sweep:
    #     wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'],
    #             name=generate_run_name(config))

    if wandb_log and no_sweep:
        wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'],
                name=run_name)


    print("MERGED CONFIGURATION FOR THIS RUN \n")
    pprint(config)
    print('\n\n')


    # The below variable tells what should be used as the feature extractor,
    # like vit, or convnext etc. It is specified in config file.
    feat_ext_name = config['model']['feat_extractor']
    feat_ext_config = config['arch'][feat_ext_name]


    # Image preprocessor is sent to AdsNonAds dataset, which will preprocess the input the same way it 
    # was done for the images during pretraining of that model.

    # NOTE: <something>FeatureExtractor == used for preprocessing purposes. 
    # IT WILL NOT RETURN feature vector of an image
    img_preprocessor = None

    if feat_ext_name == 'vit':
        img_preprocessor = ViTFeatureExtractor.from_pretrained(feat_ext_config['args']['pretrained'])
    else:
        img_preprocessor = ConvNextFeatureExtractor.from_pretrained(feat_ext_config['args']['pretrained'])

    # train_dataset = AdsNonAds(images_dirs=[config['data']['train_dir']],
    #                         img_preprocessor=img_preprocessor,
    #                         of_num_imgs=None,
    #                         overfit_test=False,
    #                         augment_data=config['data']['augment'])
    train_dataset = AdsNonAds(images_dirs=[config['data']['train_dir']],
                            img_preprocessor=img_preprocessor,
                            of_num_imgs=None,
                            overfit_test=False,
                            augment_data=config['data']['augment'])
    print(len(train_dataset))
    train_dataloader = make_data_loader(dataset=train_dataset,
                                        batch_size=config['model']['train_bs'],
                                        num_workers=4,
                                        sampler=None,
                                        data_augment=config['data']['augment'])


    valid_dataset = AdsNonAds(images_dirs=[config['data']['val_dir']],
                            img_preprocessor=img_preprocessor,
                            of_num_imgs=None,
                            overfit_test=False,
                            augment_data=False)

    valid_dataloader = make_data_loader(dataset=valid_dataset,
                                        batch_size=config['model']['valid_bs'],
                                        num_workers=4,
                                        sampler=None,
                                        data_augment=False)
    

    # We will use model_name_to_class, which is a dictionary that maps name
    # of the model, from str to class, to initialize the desired model.

    model = model_name_to_class[feat_ext_config['class']](
        pretrained = feat_ext_config['args']['pretrained'], 
        feature_dim = feat_ext_config['args']['feature_dim'] , 
        num_classes = feat_ext_config['args']['num_classes'], 
        dropout_prob = feat_ext_config['args']['dropout_prob'], 
        is_trainable = feat_ext_config['args']['is_trainable']
    )
    ckpt = torch.load(config['model']['checkpoint'])
    model.load_state_dict(ckpt["model_state_dict"])
    print("\nModel is loaded with the checkpoint")

    count_trainable_parameters(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()


    feat_ext_params = model.feat_ext.parameters()
    cls_params = model.cls_head.parameters()

    param_groups = [feat_ext_params, cls_params]
    param_grp_lr = [config['lr']['feat_ext_lr'],
                    config['lr']['base_lr']]

    # We only need to use a LR scheduler for classification head.
    has_sched = [False, True]


    optimizers, schedulers = optims_and_scheds(
        param_groups=param_groups, param_lr=param_grp_lr,
        has_sched=has_sched, config=config)


    # Some large value to start checkpointing models based on best validation loss.
    previous_valid_loss = 1e10

    epochs = config['model']['epochs']

    for epoch in range(epochs):

        train_loss, train_pred, train_gt = train_one_epoch(
            train_dataloader, criterion, model, epoch, device, optimizers, schedulers)
        
        valid_loss, valid_pred, valid_gt = valid_one_epoch(
            valid_dataloader, criterion, model, epoch, device)

        train_metrics = ["accuracy"]
        valid_metrics = ["accuracy", "f1_score"]

        train_scores = calculate_metrics(train_metrics, train_gt, train_pred)
        valid_scores = calculate_metrics(valid_metrics, valid_gt, valid_pred)

        print("train_loss:", train_loss)
        print("valid_loss:", valid_loss)
        print("Train:", train_scores)
        print("Valid:", valid_scores)
        
        if wandb_log:
            wandb.log({'loss/train': train_loss, 'loss/val': valid_loss}, step=epoch)
            wandb.log({'acc/train': train_scores['accuracy'],
                        'acc/val': valid_scores['accuracy'],
                        'f1/val': valid_scores['f1_score'],
            }, step=epoch)

        
        save_states = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers],
            'schedulers': [sch.state_dict() for sch in schedulers]
        }

        # Logging the checkpoints when the validation loss is better than the previous one
        if valid_loss < previous_valid_loss:
            previous_valid_loss = valid_loss
            save_checkpoint(state=save_states,
                            is_best=True,
                            file_folder=config['ckpt']['ckpt_folder'],
                            experiment=run_name,
                            file_name='epoch_{:03d}.pth.tar'.format(epoch)
                            )

        # Logging the checkpoints at regular checkpoint frequency
        if epoch % config['ckpt']['ckpt_frequency'] == 0:
            save_checkpoint(state=save_states, is_best=False,
                            file_folder=config['ckpt']['ckpt_folder'],
                            experiment=run_name,
                            file_name='epoch_{:03d}.pth.tar'.format(epoch)
                            )




if __name__ == '__main__':

    if config['wandb']['sweep']:
        wandb.agent(
            sweep_id=config['wandb']['sweep_id'], 
            function=sweep_agent_manager, 
            count=config['wandb']['sweep_runs']
        )
    else:
        trainer(config)
