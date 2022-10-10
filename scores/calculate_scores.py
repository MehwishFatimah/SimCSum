import os
import pandas as pd
import argparse
from evaluate_mf import Evaluate
parser = argparse.ArgumentParser(description = 'Help for the main module')
parser.add_argument('-d', type = str, default = None,   help = 'Data dir')
parser.add_argument('-sf', type = str, default = None,   help = 'source/input file name with path')
parser.add_argument('-f', type = str, default = None,   help = 'file name')


'''----------------------------------------------------------------
'''   

if __name__ == "__main__":
    

    args = parser.parse_args()
    folder = args.d
    fname =  args.f
    source = args.sf
    file = os.path.join(folder, fname)

    if os.path.exists(file):
        score = Evaluate(folder)
        score.evaluation(source, file)
                
    print("\n\n--------------------------------------------------\n\n")  