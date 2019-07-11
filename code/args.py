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

        self.parser.add_argument('--log_interval', type=int, metavar='M',
                            help='The number of batchs to wait between each console log')
        self.parser.add_argument('--num_workers', type=int,metavar='NUMWORKERS',
                            help='the number of processes to load the data. num_workers equal 0 means that it’s \
                            the main process that will do the data loading when needed, num_workers equal 1 is\
                            the same as any n, but you’ll only have a single worker, so it might be slow')
        self.parser.add_argument('--cuda', type=str2bool, metavar='S',
                            help='To run computations on the gpu')
        self.parser.add_argument('--multi_gpu', type=str2bool, metavar='S',
                            help='If cuda is true, run the computation with multiple gpu')
        self.parser.add_argument('--debug', type=str2bool,metavar='BOOL',
                            help="To run only a few batch of training and a few batch of validation")
        self.parser.add_argument('--redirect_out', type=str2bool,metavar='BIDIR',
                            help='If true, the standard output will be redirected to a file python.out')
        self.parser.add_argument('--note', type=str,metavar='NOTE',
                            help="A note on the model")

        self.parser.add_argument('--epochs', type=int, metavar='N',
                            help='maximum number of epochs to train')

        #Arg to share
        self.parser.add_argument('--model_id', type=str, metavar='IND_ID',
                            help='the id of the individual model')
        self.parser.add_argument('--exp_id', type=str, metavar='EXP_ID',
                            help='the id of the experience')
        self.parser.add_argument('--seed', type=int, metavar='S',help='Seed used to initialise the random number generator.')

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
