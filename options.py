
class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):

        # global settings
        parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')
        parser.add_argument('--arch', type=str, default='ShadowMamba', help='archtechture')
        parser.add_argument('--nepoch', type=int, default=400, help='training epochs')


        parser.add_argument('--env', type=str, default='_ISTD', help='env')
        parser.add_argument('--checkpoint', type=int, default=20, help='checkpoint')

        # Optimizer
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')


        # train
        parser.add_argument('--train_patch_size', type=int, default=320, help='patch size of training sample')
        parser.add_argument('--train_workers', type=int, default=0, help='train_dataloader workers')
        parser.add_argument('--train_dir', type=str, default='', help='dir of train data')
        parser.add_argument('--resume', action='store_true', default=True)
        parser.add_argument('--pretrain_weights', type=str, default='',
                            help='path of pretrained_weights')
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=1, help='epochs for warmup')

        # val
        parser.add_argument('--eval_workers', type=int, default=0, help='eval_dataloader workers')
        parser.add_argument('--val_dir', type=str, default='', help='dir of train data')
        parser.add_argument('--training_val_epochs', type=int, default=5, help='epochs for val')

        return parser