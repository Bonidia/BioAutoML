import argparse
import random
import warnings
warnings.filterwarnings("ignore")
from Bio import SeqIO


def sampling(finput, foutput_train, foutput_test, parameter):
    arq_train = open(foutput_train, 'a')
    arq_test = open(foutput_test, 'a')
    nameseq = {}
    for seq_record in SeqIO.parse(finput, "fasta"):
        name = seq_record.name
        seq = seq_record.seq
        nameseq[name] = seq

    keys = []
    for key, value in random.sample(nameseq.items(), parameter):
        arq_test.write(">" + key)
        keys.append(key)
        arq_test.write("\n")
        arq_test.write(str(value))
        arq_test.write("\n")

    for key, value in nameseq.items():
        if key not in keys:
            arq_train.write(">" + key)
            arq_train.write("\n")
            arq_train.write(str(value))
            arq_train.write("\n")

    return


#############################################################################    
if __name__ == "__main__":
    print("\n")
    print("###################################################################################")
    print("######################## Feature Extraction: Sampling  ############################")
    print("##############  Arguments: python3.5 -i input -o output -p samples   ##############")
    print("##########               Author: Robson Parmezan Bonidia                ###########")
    print("###################################################################################")
    print("\n")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Fasta format file, E.g., test.fasta')
    parser.add_argument('-train', '--output_train', help='Fasta format file, E.g., sampling.fasta')
    parser.add_argument('-test', '--output_test', help='Fasta format file, E.g., sampling.fasta')
    parser.add_argument('-p', '--parameter', help='Amount of Samples - Test File, e.g., 1000, 2000 ...')
    args = parser.parse_args()
    finput = str(args.input)
    foutput_train = str(args.output_train)
    foutput_test = str(args.output_test)
    parameter = int(args.parameter)
    sampling(finput, foutput_train, foutput_test, parameter)
#############################################################################
