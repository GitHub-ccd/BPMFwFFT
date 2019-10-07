import argparse
import os
import urllib

parser = argparse.ArgumentParser()
parser.add_argument("--affinity_data_dir", type=str, default="affinity")

args = parser.parse_args()

print("parsing args")
print(args.affinity_data_dir)
