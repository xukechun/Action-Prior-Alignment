import datetime
import random
import argparse
import numpy as np
import torch
from helpers.logger import Logger
from a2.train.trainer import Trainer
from helpers.data_loader import unified_data_loader, unified_adaptive_data_loader
from models.networks import CLIPAction, AdaptPolicyCLIPAction, AdaptFeatCLIPAction, CLIPLangEmbAction
from tensorboardX import SummaryWriter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size (default: 1)')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False)
    parser.add_argument('--normalize', dest='normalize', action='store_true', default=False)
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', default=False)
    parser.add_argument('--data_path', action='store', type=str, default='')
    parser.add_argument('--load_model', dest='load_model', action='store_true', default=False)
    parser.add_argument('--model_path', action='store', type=str, default='')
    parser.add_argument('--save_model_interval', type=int, default=500, metavar='N',
                        help='episode interval to save model')
    parser.add_argument('--log_suffix', action='store', type=str, default='bc')

    # Transformer Paras
    parser.add_argument('--fusion_sa', dest='fusion_sa', action='store_true', default=False)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true', default=False)
    parser.add_argument('--lang_emb', dest='lang_emb', action='store_true', default=False)
    parser.add_argument('--lang_enc', action='store', type=str, default='clip') 
    parser.add_argument('--task_emb', dest='task_emb', action='store_true', default=False)
    parser.add_argument('--agent', action='store', type=str, default='unified')
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=1) # cross attention layer
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=384, metavar='N',
                        help='hidden size (default: 384)')

    # Training Paras
    parser.add_argument('--use_rope', dest='use_rope', action='store_true', default=False)
    parser.add_argument('--no_feat_rope', dest='no_feat_rope', action='store_true', default=False)
    parser.add_argument('--no_rgb_feat', dest='no_rgb_feat', action='store_true', default=False)
    parser.add_argument('--adaptive', dest='adaptive', action='store_true', default=False)
    parser.add_argument('--adaptive_type', action='store', type=str, default='policy')
    parser.add_argument('--adaptive_way', action='store', type=str, default='residual')
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--loss', action='store', type=str, default='ce')
    parser.add_argument('--epoch_num', type=int, default=200, help='training epoch number (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate (default: 0.0003)')
    parser.add_argument('--adjust_lr', dest='adjust_lr', action='store_true', default=False)
    parser.add_argument('--step_size', type=int, default=50, help='step size of learning rate adjustment (default: step scheduler, 50)')
    parser.add_argument('--step_ratio', type=float, default=0.5, help='step ratio of learning rate adjustment (default: 0.5)')

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.task_num = 2 if args.task_emb else None

    # tensorboard setting
    tb = SummaryWriter('tensorlogs/{}_BC_PP'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    # resume logger
    if args.resume:
        args.resume_epoch = int(args.model_path.split(".")[0].split("_")[-1]) + 1
        resume_logger = os.path.dirname(os.path.dirname(args.model_path))

    # load logger
    resume_logger = resume_logger if args.resume else None
    logger = Logger(suffix=args.log_suffix, resume_logger=resume_logger)
       
    # load unified agent
    if not args.adaptive:
        if not args.lang_emb:
            agent = CLIPAction(action_dim=7, args=args)
        else:
            agent = CLIPLangEmbAction(action_dim=7, args=args)
    else:
        if args.adaptive_type == "policy":
            if args.adaptive_way == "full":
                agent = CLIPAction(action_dim=7, args=args)
            elif args.adaptive_way == "residual":
                agent = AdaptPolicyCLIPAction(action_dim=7, args=args)
            
            for k, v in agent.named_parameters():
                if args.adaptive_way == "residual":
                    v.requires_grad = False # fix parameters
                    if 'residual_policy' in k:
                        v.requires_grad = True # only the residual policy is trainable
                print(k)
                print(v.requires_grad)
                
        elif args.adaptive_type == "feat":
            agent = AdaptFeatCLIPAction(action_dim=7, args=args)
            for k, v in agent.named_parameters():
                v.requires_grad = False # fix parameters
                if 'feat_adapter' in k:
                    v.requires_grad = True # only the feature adapter is trainable
                print(k)
                print(v.requires_grad)
                        
    # save configs
    logger.save_net_arch(agent)
    logger.save_configs(vars(args))

    recorder = {'logger': logger,
                'tb': tb}

    # load resume or base model
    if args.load_model:
        logger.load_base_sl_checkpoint(agent, args.model_path, args.evaluate)
        
    # Setup data loader
    if not args.adaptive:
        train_dl = unified_data_loader(args.data_path, args.sample_num, shuffle=True)
    else:
        train_dl = unified_adaptive_data_loader(args.data_path, args.sample_num, shuffle=True)

    # Create trainer and start training
    trainer = Trainer(agent, args)
    start_epoch = args.resume_epoch if args.resume else 0
    trainer.train(train_dl, recorder, start_epoch)

if __name__ == '__main__':
    main()