from models.networks import CLIPAction, AdaptPolicyCLIPAction, AdaptFeatCLIPAction, CLIPLangEmbAction

class ViLGP3D(object):
    def __init__(self, action_dim, args):

        self.device = args.device
        self.vilg3d = CLIPAction(action_dim, args)

        if not args.evaluate:
            self.vilg3d.train()
        else:
            self.vilg3d.eval()


    def select_action(self, pts_pos, pts_feat, pts_sim, actions):
        logits, action = self.vilg3d(pts_pos, pts_feat, pts_sim, actions)

        return logits.detach().cpu().numpy(), action.detach().cpu().numpy()[0]


class LangEmbViLGP3D(object):
    def __init__(self, action_dim, args):

        self.device = args.device
        self.vilg3d = CLIPLangEmbAction(action_dim, args)

        if not args.evaluate:
            self.vilg3d.train()
        else:
            self.vilg3d.eval()


    def select_action(self, pts_pos, pts_feat, actions, lang_goal):
        logits, action = self.vilg3d(pts_pos, pts_feat, actions, lang_goal)

        return logits.detach().cpu().numpy(), action.detach().cpu().numpy()[0]
    

class AdaptViLGP3D(object):
    def __init__(self, action_dim, args):

        self.device = args.device
        if args.adaptive_type == "policy":
            self.vilg3d = AdaptPolicyCLIPAction(action_dim, args)
        elif args.adaptive_type == "feat":
            self.vilg3d = AdaptFeatCLIPAction(action_dim, args)

        if not args.evaluate:
            self.vilg3d.train()
        else:
            self.vilg3d.eval()


    def select_action(self, pts_pos, pts_feat, pts_sim, actions, ratio=0.2):
        logits, action = self.vilg3d(pts_pos, pts_feat, pts_sim, actions, ratio=ratio)

        return logits.detach().cpu().numpy(), action.detach().cpu().numpy()[0]