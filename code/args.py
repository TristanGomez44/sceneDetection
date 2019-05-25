import sys
import argparse
import configparser

def str2bool(v):
    '''Convert a string to a boolean value'''
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2FloatList(x):

    '''Convert a formated string to a list of float value'''
    if len(x.split(",")) == 1:
        return float(x)
    else:
        return [float(elem) for elem in x.split(",")]
def strToStrList(x):
    if x == "None":
        return []
    else:
        return x.split(",")

def str2StrList(x):
    '''Convert a string to a list of string value'''
    return x.split(" ")

class ArgReader():
    """
    This class build a namespace by reading arguments in both a config file
    and the command line.

    If an argument exists in both, the value in the command line overwrites
    the value in the config file

    This class mainly comes from :
    https://stackoverflow.com/questions/3609852/which-is-the-best-way-to-allow-configuration-options-be-overridden-at-the-comman
    Consulted the 18/11/2018

    """

    def __init__(self,argv):
        ''' Defines the arguments used in several scripts of this project.
        It reads them from a config file
        and also add the arguments found in command line.

        If an argument exists in both, the value in the command line overwrites
        the value in the config file
        '''

        # Do argv default this way, as doing it in the functional
        # declaration sets it at compile time.
        if argv is None:
            argv = sys.argv

        # Parse any conf_file specification
        # We make this parser with add_help=False so that
        # it doesn't parse -h and print help.
        conf_parser = argparse.ArgumentParser(
            description=__doc__, # printed with -h/--help
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # Turn off help, so we print all options in response to -h
            add_help=False
            )
        conf_parser.add_argument("-c", "--conf_file",
                            help="Specify config file", metavar="FILE")
        args, self.remaining_argv = conf_parser.parse_known_args()

        defaults = {}

        if args.conf_file:
            config = configparser.SafeConfigParser()
            config.read([args.conf_file])
            defaults.update(dict(config.items("default")))

        # Parse rest of arguments
        # Don't suppress add_help here so it will handle -h
        self.parser = argparse.ArgumentParser(
            # Inherit options from config_parser
            parents=[conf_parser]
            )
        self.parser.set_defaults(**defaults)

        # Training settings
        #parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

        self.parser.add_argument('--log_interval', type=int, metavar='M',
                            help='The number of batchs to wait between each console log')

        self.parser.add_argument('--nb_worse_epochs', type=float, metavar='M',
                            help='The number of epochs during which the performance on validation set can decrease without the training stops.')
        self.parser.add_argument('--epochs', type=int, metavar='N',
                            help='maximum number of epochs to train')

        self.parser.add_argument('--feat', type=str, metavar='N',
                            help='the net to use to produce feature for each shot')

        self.parser.add_argument('--feat_audio', type=str, metavar='N',
                            help='the net to use to produce audio feature for each shot')

        self.parser.add_argument('--frames_per_shot', type=int, metavar='N',
                            help='The number of frame to use to represent each shot')
        self.parser.add_argument('--frame_att_rep_size', type=int, metavar='N',
                            help='The size of the internal representation of the frame attention model')

        self.parser.add_argument('--pretrain_dataset', type=str, metavar='N',
                            help='The network producing the features can be either pretrained on \'imageNet\' or \'places365\'. This argument \
                                selects one of the two datasets.')
        self.parser.add_argument('--hidden_size', type=int,metavar='HS',
                            help='The size of the hidden layers in the RNN')
        self.parser.add_argument('--num_layers', type=int,metavar='NL',
                            help='The number of hidden layers in the RNN')
        self.parser.add_argument('--dropout', type=float,metavar='D',
                            help='The dropout amount on each layer of the RNN except the last one')
        self.parser.add_argument('--bidirect', type=str2bool,metavar='BIDIR',
                            help='If true, the RNN will be bi-bidirectional')
        self.parser.add_argument('--redirect_out', type=str2bool,metavar='BIDIR',
                            help='If true, the standard output will be redirected to a file python.out')

        self.parser.add_argument('--train_visual', type=str2bool,metavar='BOOL',
                            help='If true, the visual feature extractor will also be trained')

        self.parser.add_argument('--train_audio', type=str2bool,metavar='BOOL',
                            help='If true, the audio feature extractor will also be trained')

        self.parser.add_argument('--lr', type=str2FloatList,metavar='LR',
                            help='learning rate (it can be a schedule : --lr 0.01,0.001,0.0001)')
        self.parser.add_argument('--batch_size', type=int,metavar='BS',
                            help='The batchsize to use for training')

        self.parser.add_argument('--val_batch_size', type=int,metavar='BS',
                            help='The batchsize to use for validation')

        self.parser.add_argument('--l_min', type=int,metavar='LMIN',
                            help='The minimum length of a training sequence')
        self.parser.add_argument('--l_max', type=int,metavar='LMAX',
                            help='The maximum length of a training sequence')

        self.parser.add_argument('--val_l', type=int,metavar='LMAX',
                            help='Length of sequences for validation')

        self.parser.add_argument('--chan_temp_mod', type=int,metavar='LMAX',
                            help='The channel number of the temporal model, if it is a CNN')

        self.parser.add_argument('--pretr_temp_mod', type=str2bool, metavar='S',
                            help='To have the temporal model pretrained on ImageNet, if it is a CNN')

        self.parser.add_argument('--lay_feat_cut', type=int,metavar='LMAX',
                            help='The layer at which to take the feature in case which the resnet feature extractor is chosen.')

        self.parser.add_argument('--img_width', type=int,metavar='WIDTH',
                            help='The width of the resized images, if resize_image is True, else, the size of the image')
        self.parser.add_argument('--img_heigth', type=int,metavar='HEIGTH',
                            help='The height of the resized images, if resize_image is True, else, the size of the image')

        self.parser.add_argument('--train_part_beg', type=float,metavar='START',
                            help='The (normalized) start position of the dataset to use for training')
        self.parser.add_argument('--train_part_end', type=float,metavar='END',
                            help='The (normalized) end position of the dataset to use for training')

        self.parser.add_argument('--val_part_beg', type=float,metavar='START',
                            help='The (normalized) start position of the dataset to use for validation')
        self.parser.add_argument('--val_part_end', type=float,metavar='END',
                            help='The (normalized) end position of the dataset to use for validation')

        self.parser.add_argument('--test_part_beg', type=float,metavar='START',
                            help='The (normalized) start position of the dataset to use for testing')
        self.parser.add_argument('--test_part_end', type=float,metavar='END',
                            help='The (normalized) end position of the dataset to use for testing')

        self.parser.add_argument('--num_workers', type=int,metavar='NUMWORKERS',
                            help='the number of processes to load the data. num_workers equal 0 means that it’s \
                            the main process that will do the data loading when needed, num_workers equal 1 is\
                            the same as any n, but you’ll only have a single worker, so it might be slow')
        self.parser.add_argument('--momentum', type=float, metavar='M',
                            help='SGD momentum')
        self.parser.add_argument('--seed', type=int, metavar='S',
                            help='Seed used to initialise the random number generator.')

        self.parser.add_argument('--model_id', type=str, metavar='IND_ID',
                            help='the id of the individual model')
        self.parser.add_argument('--exp_id', type=str, metavar='EXP_ID',
                            help='the id of the experience')

        self.parser.add_argument('--dataset_train', type=str, metavar='N',help='the dataset to train. Can be \'OVSD\', \'PlanetEarth\' or \'RAIDataset\'.')
        self.parser.add_argument('--dataset_val', type=str, metavar='N',help='the dataset to validate. Can be \'OVSD\', \'PlanetEarth\' or \'RAIDataset\'.')
        self.parser.add_argument('--dataset_test', type=str, metavar='N',help='the dataset to testing. Can be \'OVSD\', \'PlanetEarth\' or \'RAIDataset\'.')

        self.parser.add_argument('--cuda', type=str2bool, metavar='S',
                            help='To run computations on the gpu')

        self.parser.add_argument('--resize_image', type=str2bool, metavar='S',
                            help='to resize the image to the size indicated by the img_width and img_heigth arguments.')

        self.parser.add_argument('--multi_gpu', type=str2bool, metavar='S',
                            help='If cuda is true, run the computation with multiple gpu')

        self.parser.add_argument('--class_weight', type=float, metavar='S',
                            help='Set the importance of balancing according to class instance number in the loss function. 0 makes equal weights and 1 \
                            makes weights proportional to the class instance number of the other class.')

        self.parser.add_argument('--optim', type=str, metavar='OPTIM',
                            help='the optimizer to use (default: \'SGD\')')

        self.parser.add_argument('--start_mode', type=str,metavar='SM',
                    help='The mode to use to initialise the model. Can be \'scratch\' or \'fine_tune\'.')

        self.parser.add_argument('--init_path', type=str,metavar='SM',
                    help='The path to the weight file to use to initialise the network')
        self.parser.add_argument('--init_path_visual', type=str,metavar='SM',
                    help='The path to the weight file to use to initialise the visual model')
        self.parser.add_argument('--init_path_audio', type=str,metavar='SM',
                    help='The path to the weight file to use to initialise the audio model')


        self.parser.add_argument('--noise', type=float, metavar='NOISE',
                    help='the amount of noise to add in the gradient of the model (as a percentage of the norm)(default: 0.1)')

        self.parser.add_argument('--param_to_opti', type=strToStrList,metavar='V',
                            help="The parameters to optimise. Can be a list with elements among 'RNN','CNN'")

        self.parser.add_argument('--note', type=str,metavar='NOTE',
                            help="A note on the model")

        self.parser.add_argument('--audio_len', type=float,metavar='NOTE',
                            help="The length of the audio for each shot (in seconds)")
        self.parser.add_argument('--margin', type=float,metavar='NOTE',
                            help="The margin for the siamese network training.")
        self.parser.add_argument('--dist_order', type=int,metavar='NOTE',
                            help="The distance order to measure similarity for the siamese network.")
        self.parser.add_argument('--mining_mode', type=str,metavar='MODE',
                            help="The mining mode to use to train the siamese net. Can only be \'offline\'.")

        self.parser.add_argument('--soft_loss', type=str2bool,metavar='MODE',
                            help="To use target soften with a triangular kernel.")
        self.parser.add_argument('--soft_loss_width', type=str2FloatList,metavar='MODE',
                            help="The width of the triangular window of the soft loss (in number of shots). Can be a schedule like learning rate")

        self.parser.add_argument('--temp_model', type=str,metavar='MODE',
                            help="The architecture to use to model the temporal dependencies. Can be \'RNN\', \'resnet50\' or \'resnet101\'.")
        self.parser.add_argument('--debug', type=str2bool,metavar='BOOL',
                            help="To run only a few batch of training and a few batch of validation")

        self.args = None

    def getRemainingArgs(self):
        ''' Reads the comand line arg'''

        self.args = self.parser.parse_args(self.remaining_argv)

    def writeConfigFile(self,filePath):
        """ Writes a config file containing all the arguments and their values"""

        config = configparser.SafeConfigParser()
        config.add_section('default')

        for k, v in  vars(self.args).items():
            config.set('default', k, str(v))

        with open(filePath, 'w') as f:
            config.write(f)
