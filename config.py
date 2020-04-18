import argparse


parser = argparse.ArgumentParser(description='Hyper-parameters management')


# network options
parser.add_argument('--num_classes', type=int, default=4, help='the number of classes')
parser.add_argument('--in_channels', type=int, default=2, help='channel of Network input')
parser.add_argument('--num_filters', type=int, default=32, help='number of filters for initial_conv')

args = parser.parse_args()


