import argparse
from a2.evaluate.evaluator import PickPlaceEvaluator

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')

    parser.add_argument('--gui', dest='gui', action='store_true', default=False)
    parser.add_argument('--unseen', dest='unseen', action='store_true', default=False)
    parser.add_argument('--direct_grounding', dest='direct_grounding', action='store_true', default=False)
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', default=True)
    parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
    parser.add_argument('--sample_grasp', dest='sample_grasp', action='store_true', default=False)
    parser.add_argument('--sample_place', dest='sample_place', action='store_true', default=False)
    parser.add_argument('--workspace', action='store', type=str, default='extend')
    parser.add_argument('--testing_case_dir', action='store', type=str, default='testing_cases/')
    parser.add_argument('--testing_case', action='store', type=str, default=None)
    parser.add_argument('--log_suffix', action='store', type=str, default=None)
    parser.add_argument('--load_model', dest='load_model', action='store_true', default=False)
    parser.add_argument('--model_path', action='store', type=str, default='')

    parser.add_argument('--single_view', dest='single_view', action='store_true', default=False)
    parser.add_argument('--diff_views', dest='diff_views', action='store_true', default=False)
    parser.add_argument('--mode', action='store', type=str, default="pickplace") # grasp, pickplace
    parser.add_argument('--mb_grasp', dest='mb_grasp', action='store_true', default=False)
    parser.add_argument('--sample_num', action='store', type=int, default=500)
    parser.add_argument('--num_episode', action='store', type=int, default=15)
    parser.add_argument('--max_episode_step', type=int, default=8)
    parser.add_argument('--action_var', dest='action_var', action='store_true', default=False)

    # Transformer paras
    parser.add_argument('--ratio', action='store', type=float, default=0.2)
    parser.add_argument('--feat_backbone', action='store', type=str, default='clip')
    parser.add_argument('--fusion_sa', dest='fusion_sa', action='store_true', default=False) 
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true', default=False) 
    parser.add_argument('--lang_emb', dest='lang_emb', action='store_true', default=False)
    parser.add_argument('--lang_enc', action='store', type=str, default='longclip')
    parser.add_argument('--use_rope', dest='use_rope', action='store_true', default=False)
    parser.add_argument('--no_feat_rope', dest='no_feat_rope', action='store_true', default=False)
    parser.add_argument('--no_rgb_feat', dest='no_rgb_feat', action='store_true', default=False)
    parser.add_argument('--normalize', dest='normalize', action='store_true', default=False)
    parser.add_argument('--workspace_shift', dest='workspace_shift', action='store_true', default=False)
    parser.add_argument('--adaptive', dest='adaptive', action='store_true', default=False)
    parser.add_argument('--adaptive_type', action='store', type=str, default='policy')
    parser.add_argument('--task_emb', dest='task_emb', action='store_true', default=False)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=1) # cross attention layer
    parser.add_argument('--heads', type=int, default=8)

    # SAC parameters
    parser.add_argument('--hidden_size', type=int, default=384, metavar='N',
                        help='hidden size (default: 384)')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluator = PickPlaceEvaluator(args)
    evaluator.evaluate()