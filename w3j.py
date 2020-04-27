from optparse import OptionParser
import utils as ut

parser = OptionParser()
parser.add_option('--lmax', dest='ellmax',  default=383, type=int,
                  help='Set to define lmax for w3j, default=383')
(o, args) = parser.parse_args()

# Compute the Wigner 3J symbols if they haven't yet calculated
ut.get_w3j(lmax=o.ellmax)

