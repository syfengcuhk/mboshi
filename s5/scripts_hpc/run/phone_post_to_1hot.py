import sys
#import numpy as np

# This code runs with 1 input. Input 1 denotes file for input, i.e. the Kaldi phone_post format
# USed to convert phone posteriorgram to 1hot.

file_read = sys.argv[1]
file_write = sys.argv[2]  #file_read + ".1hot" 
print("Starting to process file %s  " % (file_read))
with open(file_read, 'r') as f:
    content = f.readlines()
    with open(file_write, 'w') as f_w:
        for this_line in content:
            # Every line starts with uttid and followed by many [ phone_id post phone_id post ...] [ phone_id post phone_id post ... ]
            uttid = this_line.split()[0]
#            original_post = this_line.split()[1:] # " [ XX XX XX XX ] [ XX XX XX XX XX XX ] ... [ XX XX ] "
            original_post = this_line.split('[ ')[1:] # original_post[0] = ' XX XX XX XX ] '
            phone_this_line = [] 
            for frame_index in range(len(original_post)):
                this_frame = original_post[frame_index].split()[:-1]
                phone_id = this_frame[0::2]
                post_val = this_frame[1::2]
                phone_this_frame = phone_id[post_val.index(max(post_val))]
                phone_this_line.append(phone_this_frame)
            f_w.write(uttid + " " + " ".join(phone_this_line) + "\n")
            #print("uttid: %s; orignial: %s" % (uttid, original_post) )
