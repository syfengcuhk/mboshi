import sys

# This code runs with 2 inputs. Input 1 denotes file for input; Input 2 is a integer denoting number of times of repetition
# USed to convert subsampled forced alignments to normal alignments.
# Subsampled forced alignments are usually generated from nnet3-chain AM, with subsampling factor 3.

file_read = sys.argv[1]
rep_times = sys.argv[2]
file_write = file_read + ".rep" + str(rep_times)
print("Starting to process file %s by factor %s " % (file_read, rep_times ))
with open(file_read, 'r') as f:
    content = f.readlines()
    with open(file_write, 'w') as f_w:
        for this_line in content:
            uttid = this_line.split()[0]
            original_align = this_line.split()[1:]
            #print("uttid: %s; orignial: %s" % (uttid, original_align) )
            rep_align = [ item for item in original_align for n in range(int(rep_times))  ]
            f_w.write(uttid + " " + " ".join(rep_align) + "\n")
#            for n in range(rep_times):
#                f_w.write(origin)
#        f.write()
