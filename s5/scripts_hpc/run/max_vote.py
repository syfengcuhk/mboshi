from __future__ import division
import sys
import os
# receives as input two alignment files, a threshold ranging between [0:1], and output one file
# threshold larger: more strict to uniform labels. 1: not vote; 0: always vote. 0.5: if max_vote more than 50% of the frames within segments, uniform
old_aud_in = sys.argv[1]
ref_segmentation = sys.argv[2]
max_vote_thres = float(sys.argv[3])
print("%s: old alignment file %s, reference file %s" % ( sys.argv[0], old_aud_in , ref_segmentation))

with open(ref_segmentation, 'r') as f_ref:
    uttid = []
    master_phone_boundary = []
    ref_segment = f_ref.readlines()
    for this_line in ref_segment:
        this_line_split = this_line.strip().split()
        uttid.append(this_line_split[0])
        this_phone_boundary = [ n-1 for n in range(2,len(this_line_split)) if this_line_split[n] != this_line_split[n-1]  ]
        master_phone_boundary.append(this_phone_boundary)

#print("length of master_phone_boundary: %d" % ( len(master_phone_boundary) ))
print(master_phone_boundary[0])

with open(old_aud_in, 'r') as f_old_aud_in:
    uttid_old_aud_in = []
    master_alignment_old = []
    ali_old = f_old_aud_in.readlines()
    for this_line in ali_old:
        this_line_split = this_line.strip().split()
        uttid_old_aud_in.append(this_line_split[0])
        master_alignment_old.append(this_line_split[1:])

if uttid_old_aud_in != uttid:
    print("input1 and input2 have different line order")
    sys.exit(0)

#print(master_alignment_old[0])
with open(old_aud_in + '_maxvote' + str(max_vote_thres) , 'w') as f_w:
    master_alignment_new = []
    for utt_index in range(len(uttid)):
        align_new_this_utt = []
        phone_boundary_this_utt = master_phone_boundary[utt_index]
    #    [0:phone_boundary_this_utt[0]], [phone_boundary_this_utt[0]: phone_boundary_this_utt[1]] ...
        alignment_this_utt = master_alignment_old[utt_index]
        most_label_this_segment = max(set(alignment_this_utt[0:phone_boundary_this_utt[0]]  ), key = alignment_this_utt[0:phone_boundary_this_utt[0]].count)
        if alignment_this_utt[0:phone_boundary_this_utt[0]].count(most_label_this_segment) / phone_boundary_this_utt[0] >= max_vote_thres: 
            align_new_this_utt.extend( [ most_label_this_segment ] * (phone_boundary_this_utt[0]) )
        else:
            align_new_this_utt.extend( alignment_this_utt[0:phone_boundary_this_utt[0]] )
        for segment_index in range( len(phone_boundary_this_utt) -1 ):
            most_label_this_segment = max( set( alignment_this_utt[phone_boundary_this_utt[segment_index]: phone_boundary_this_utt[segment_index + 1]] ), key = alignment_this_utt[phone_boundary_this_utt[segment_index]: phone_boundary_this_utt[segment_index + 1]].count )
            if alignment_this_utt[phone_boundary_this_utt[segment_index]: phone_boundary_this_utt[segment_index + 1]].count(most_label_this_segment) / ( phone_boundary_this_utt[segment_index + 1] - phone_boundary_this_utt[segment_index] ) >= max_vote_thres:
                align_new_this_utt.extend( [ most_label_this_segment ] * ( phone_boundary_this_utt[segment_index + 1] - phone_boundary_this_utt[segment_index] ) )
            else:
                align_new_this_utt.extend(  alignment_this_utt[phone_boundary_this_utt[segment_index]: phone_boundary_this_utt[segment_index + 1]] )
        # now comes to the last segment
        most_label_this_segment = max(set(alignment_this_utt[phone_boundary_this_utt[-1]:]  ), key = alignment_this_utt[phone_boundary_this_utt[-1]:].count)
        if alignment_this_utt[phone_boundary_this_utt[-1]:].count(most_label_this_segment) / len(alignment_this_utt[phone_boundary_this_utt[-1]:]) >= max_vote_thres:
            align_new_this_utt.extend( [ most_label_this_segment ] * len(alignment_this_utt[phone_boundary_this_utt[-1]:]) )
        else:
            align_new_this_utt.extend(  alignment_this_utt[phone_boundary_this_utt[-1]:] )
        f_w.write(uttid[utt_index] + ' ' + " ".join(align_new_this_utt) + "\n")


