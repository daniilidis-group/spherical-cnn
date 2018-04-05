import argparse
import os

import yaml


def parse(args=None):
    parser = argparse.ArgumentParser(description='Train a model.',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--model', '-m', type=str, default='',
                        help='model name (a function with the same name must exist in models.py)')
    parser.add_argument('--logdir', '-ld', type=str, default='',
                        help='directory to save models, logs and checkpoints.')
    parser.add_argument('--run_id', '-id', type=str, default='',
                        help='identifier for this run')
    parser.add_argument('--test_only', '-to', action='store_true', default=False,
                        help='only run test')

    # dataset params
    parser.add_argument('--dset', '-d', type=str, default='',
                        help='dataset loader')
    parser.add_argument('--dset_dir', '-dd', type=str, default=os.path.expanduser('~/bigdata'),
                        help='datasers directory')
    parser.add_argument('--n_classes', '-nc', type=int, default=10,
                        help='number of classes in dataset')
    parser.add_argument('--dtype', '-dt', type=str, default='float32',
                        help='dataset type')
    parser.add_argument('--spectral_input', action='store_true', default=False,
                        help='Inputs in spectral domain (2, b, b) instead of spatial (2b, 2b)')
    parser.add_argument('--mag_ang', action='store_true', default=False,
                        help='take two channels: magnitude and phase')
    parser.add_argument('--first_n', '-fn', type=int, default=None,
                        help='use only first n entries/batches for train')
    parser.add_argument('--test_n', '-tn', type=int, default=0,
                        help='use only first n entries/batches for test')
    parser.add_argument('--input_res', '-res', type=int, default=32,
                        help='resolution for spherical inputs; may subsample if larger')
    parser.add_argument('--skip_val', '-sv', action='store_true', default=False,
                        help='skip validation')

    # training
    parser.add_argument('--optimizer', '-o', type=str, default='adam',
                        choices=['adam', 'momentum'],
                        help='optimizer to use')
    parser.add_argument('--learning_rate', '-lr', type=yaml.load, default={0: 1e-3},
                        help='learning rate; given as dict of form {epoch: value}')
    parser.add_argument('--regularization_weight', '-reg', type=float, default=0,
                        help='regularization loss weight')
    parser.add_argument('--n_epochs', '-ne', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--from_scratch', '-fs', action='store_true', default=False,
                        help='train from scratch (remove any previous checkpoints in logdir)')
    parser.add_argument('--train_bsize', '-bs', type=int, default=128,
                        help='training batch size')

    # model params
    parser.add_argument('--nchannels', default=1, type=int,
                        help='Number of input channels')
    parser.add_argument('--nfilters', default=[16, 16, 32, 32, 64, 64],
                        type=lambda x: [int(_) for _ in x.split(',')],
                        help='Number of filters per layer')
    parser.add_argument('--pool_layers', default=[0, 1, 0, 1, 0, 0],
                        type=lambda x: [int(_) for _ in x.split(',')],
                        help='Pooling layer indicator')
    parser.add_argument('--concat_branches', default=[0, 0, 0, 0, 0, 0],
                        type=lambda x: [int(_) for _ in x.split(',')],
                        help='Which layers to concatenate, when using a two-branch network')
    parser.add_argument('--dropout', '-do', action='store_true', default=False,
                        help='Use dropout, where applicable')
    parser.add_argument('--batch_norm', '-bn', action='store_true', default=False,
                        help='Use batch normalization')
    parser.add_argument('--batch_renorm', '-brn', action='store_true', default=False,
                        help='Use batch re-normalization (only if batch_norm == True)')
    parser.add_argument('--nonlin', '-nl', type=str, default='prelu',
                        help='Nonlinearity to be used')
    parser.add_argument('--spectral_pool', '-sp', action='store_true', default=False,
                        help='Use spectral pooling instead of max-pooling. ')
    parser.add_argument('--pool', '-p', choices=['wap', 'max', 'avg'], default='wap',
                        help='Type of pooling.')
    parser.add_argument('--n_filter_params', '-nfp', type=int, default=0,
                        help='Number of filter params (if 0, use max, else do spectral linear interpolation for localized filters.)')
    parser.add_argument('--weighted_sph_avg', '-wsa', action='store_true', default=False,
                        help='Use sin(lat) to weight averages on sphere.')
    parser.add_argument('--final_pool', '-fp', choices=['gap', 'max', 'magnitudes', 'all'], default='gap',
                        help='Final pooling layer: GAP, MAX, or frequency magnitudes?')
    parser.add_argument('--extra_loss', '-el', action='store_true', default=False,
                        help='Add extra loss on second branch for two-branch architecture. ')
    parser.add_argument('--triplet_loss', '-tl', action='store_true', default=False,
                        help='Use within-batch triplet loss for retrieval')

    parser.add_argument('--round_batches', '-rb', action='store_true', default=False,
                        help='Make sure batches always have the same size; necessary for triplet_loss')

    parser.add_argument('--no_final_fc', '-nofc', action='store_true', default=False,
                        help='Do not use a final fully connected layer.')
    parser.add_argument('--transform_method', '-tm', choices=['naive', 'sep'], default='naive',
                        help='SH transform method: NAIVE or SEParation of variables')
    parser.add_argument('--real_inputs', '-ri', action='store_true', default=False,
                        help='Leverage symmetry when inputs are real.')

    # use given args instead of cmd line, if they exist
    if isinstance(args, list):
        # if list, parse as cmd lines arguments
        args_out = parser.parse_args(args)
    elif args is not None:
        # if dict, set values directly
        args_out = parser.parse_args('')
        for k, v in args.items():
            setattr(args_out, k, v)
    else:
        args_out = parser.parse_args()

    args_out.logdir = os.path.expanduser(args_out.logdir)
    args_out.dset = os.path.expanduser(args_out.dset)
    args_out.dset_dir = os.path.expanduser(args_out.dset_dir)
    if not args_out.run_id:
        args_out.run_id = os.path.split(args_out.logdir)[-1]

    check(args_out, parser)

    return args_out


def check(args, parser):
    assert len(args.nfilters) == len(args.pool_layers)
    assert os.path.isdir(args.dset_dir)

    if args.test_only:
        assert not args.from_scratch

    if args.model == 'two_branch':
        assert len(args.nfilters) == len(args.concat_branches)

    if args.extra_loss:
        assert args.model == 'two_branch'

    if args.triplet_loss:
        assert args.round_batches

    if args.spectral_pool:
        # should not change default pooling, as it will not be used!
        assert args.pool == 'wap'
