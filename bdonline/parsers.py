def parse_command_line_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description="""Launch server for online decoding.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # see http://stackoverflow.com/a/24181138/1469195
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('--fs', action='store', type=int,
                                help="Sampling rate of EEG signal (in Hz). Only used to convert "
                                     "other arguments from milliseconds to number of samples", required=True)
    required_named.add_argument('--expfolder', action='store',
                                help='Folder with model etc.', required=True)
    required_named.add_argument('--inputsamples', action='store',
                                type=int,
                                help='Input samples (!) for the ConvNet (in samples!).', required=True)
    parser.add_argument('--inport', action='store', type=int,
                        default=7987, help='Port from which to accept incoming sensor data.')
    parser.add_argument('--outhost', action='store',
                        default='172.30.0.117', help='Hostname/IP of the prediction receiver')
    parser.add_argument('--outport', action='store', type=int,
                        default=30000, help='Port of the prediction receiver')
    parser.add_argument('--paramsfile', action='store',
                        help='Use these (possibly adapted) parameters for the model. '
                             'Filename should end with model_params.npy. Can also use "newest"'
                             'to load the newest available  parameter file. '
                             'None means to not load any new parameters, instead use '
                             'originally (offline)-trained parameters.')
    parser.add_argument('--plot', action='store_true',
                        help="Show plots of the sensors first.")
    parser.add_argument('--noout', action='store_true',
                        help="Don't wait for prediction receiver.")
    parser.add_argument('--noadapt', action='store_true',
                        help="Don't adapt model while running online.")
    parser.add_argument('--updatesperbreak', action='store', default=5,
                        type=int, help="How many updates to adapt the model during trial break.")
    parser.add_argument('--batchsize', action='store', default=45, type=int,
                        help="Batch size for adaptation updates.")
    parser.add_argument('--learningrate', action='store', default=1e-4,
                        type=float, help="Learning rate for adaptation updates.")
    parser.add_argument('--mintrials', action='store', default=10, type=int,
                        help="Number of trials before starting adaptation updates.")
    parser.add_argument('--trialstartoffsetms', action='store', default=500, type=int,
                        help="Time offset for the first sample to use (within a trial, in ms) "
                             "for adaptation updates.")
    parser.add_argument('--breakstartoffsetms', action='store', default=1000, type=int,
                        help="Time offset for the first sample to use (within a break(!), in ms) "
                             "for adaptation updates.")
    parser.add_argument('--breakstopoffsetms', action='store', default=-1000, type=int,
                        help="Sample offset for the last sample to use (within a break(!), in ms) "
                             "for adaptation updates.")
    parser.add_argument('--predgap', action='store', default=200, type=int,
                        help="Amount of milliseconds between predictions.")
    parser.add_argument('--minbreakms', action='store', default=2000, type=int,
                        help="Minimum length of a break to be used for training (in ms).")
    parser.add_argument('--mintrialms', action='store', default=0, type=int,
                        help="Minimum length of a trial to be used for training (in ms).")
    parser.add_argument('--noprint', action='store_true',
                        help="Don't print on terminal.")
    parser.add_argument('--nosave', action='store_true',
                        help="Don't save streamed data (including markers).")
    parser.add_argument('--noolddata', action='store_true',
                        help="Dont load and use old data for adaptation")
    parser.add_argument('--plotbackend', action='store',
                        default='agg', help='Matplotlib backend to use for plotting.')
    parser.add_argument('--nooldadamparams', action='store_true',
                        help='Do not load old adam params.')
    parser.add_argument('--nobreaktraining', action='store_true',
                        help='Do not use the breaks as training examples for the rest class.')
    parser.add_argument('--cpu', action='store_true',
                        help='Use the CPU instead of GPU/Cuda.')
    parser.add_argument('--nchans', action='store', default=64, type=int,
                        help="Number of EEG channels")
    args = parser.parse_args()
    assert args.breakstopoffsetms <= 0, ("Please supply a nonpositive break stop "
                                         "offset, you supplied {:d}".format(args.breakstopoffset))
    return args
