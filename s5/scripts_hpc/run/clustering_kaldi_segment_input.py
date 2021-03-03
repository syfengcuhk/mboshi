# segment level clustering, segment boundary received as the last parameter
import sys
import os
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
import hdbscan
import numpy as np

# input1 kaldi-format (text-format, not binary) feats.ark
# input2: int, num_jobs
# input3: int, random state
# input4: int, number of final clusters
file_in = sys.argv[1] #+  "debug_small"
num_jobs = int(sys.argv[2])
rand_state = int(sys.argv[3])
nclusters = int(sys.argv[4])
algorithm = sys.argv[5] # kmeans, spectral, dbscan
ref_segmentation = sys.argv[6]
subsampling_flag = int(sys.argv[7]) # if <=1 and >=-1,  do nothing; if == 2, segment represented [1st-half-mean 2nd-half-mean], if == 3, [1st-third-mean, 2nd-third-mean, 3rd-third-mean]; if == -3 [whole-mean, 1st-half-mean, 2nd-half-mean]; (3Mar) if >3, then (need pad) [ 1/subsampling_flag, 2/subsampling_flag, ..., subsampling_flag/subsampling_flag  ] i.e. generalization to == 2 and ==3
# Debug purpose
#file_in = "exp/chain_discophone_lat_label/universal_phn_ug_multi/lat_gen_acwt10.0/tdnn1g_lr2.5e-4_full_own_gmm/bnf_prefinal_l_28/full/feats.ark.txt"
#num_jobs = 1
#rand_state = 0
#nclusters = 50
#ref_segmentation = "exp/gmm_discophone_label_collapse_ali/universal_phn_ug_multi/full/tri5_phone_ali_retry_beam1280/phone_ali.ali" 
#algorithm = "kmeans"
print("%s: segmentation boundary from %s" % (sys.argv[0], ref_segmentation  ))
print("Trying to read %s" % file_in)
print("Output file: %s" % os.path.join(os.path.dirname(file_in), 'output_' + os.path.basename(file_in) + '_seg_clusters' + str(nclusters)  +  '_rand_' + str(rand_state)) )
with open(ref_segmentation, 'r') as f_ref:
    uttid = []
    master_phone_boundary = []
    ref_segment = f_ref.readlines()
    for this_line in ref_segment:
        this_line_split = this_line.strip().split()
        uttid.append(this_line_split[0])
        this_phone_boundary = [ n-1 for n in range(2,len(this_line_split)) if this_line_split[n] != this_line_split[n-1]  ]
        master_phone_boundary.append(this_phone_boundary)

print("master_phone_boundary size: %d lines" % len(master_phone_boundary))
#print(master_phone_boundary[0])
with open(file_in, 'r') as f_in:
    uttid = []
    nframes_per_utt = []
    input_data = f_in.readlines()
    master_feats = []
    print("Opend file %s, %d lines" % ( file_in, len(input_data) ))
    for this_line in input_data:
        this_line_splitted = this_line.strip().split()
        if this_line_splitted[-1] == '[':
            uttid.append(this_line_splitted[0])
            nframes_this_utt = 0
        else:
            nframes_this_utt += 1
            if this_line_splitted[-1] == ']' :
                nframes_per_utt.append(nframes_this_utt)
                master_feats.append(this_line_splitted[:-1])
            else:
                master_feats.append(this_line_splitted)
if len(uttid) != len(nframes_per_utt):
    print("uttid length != nframes_per_utt length, check input again")
#elif len(uttid) != len(master_phone_boundary): # comment for debug purpose
#    print("uttid length != master_phone_boundary length, check again")
else:
    print("Frame Master feats: %d-by-%d matrix" % ( len(master_feats), len(master_feats[0]) ))
    print("nframes_per_utt (first 10 utterances): " + str(nframes_per_utt[0:10]))
    print("Total no. frames: sum(nframes_per_utt): %d ; \nTotal no. utterances: len(uttid): %d" % ( sum(nframes_per_utt) , len(uttid)))
    # convert frame level feature to segment level
    # for now use average over frames within segments
    master_seg_feats = []
    nsegments_per_utt = [0] * len(nframes_per_utt)
    pos = 0
    for uttindex in range(len(nframes_per_utt)):
        complete_phone_boundary_this_seg =  master_phone_boundary[uttindex] #str(nframes_per_utt[uttid])
        complete_phone_boundary_this_seg.append(nframes_per_utt[uttindex])
        complete_phone_boundary_this_seg.insert(0, 0)
#        map(int, complete_phone_boundary_this_seg)
        #print(complete_phone_boundary_this_seg)
        # every segment is [complete_phone_boundary_this_seg[0]:complete_phone_boundary_this_seg[1]]
        for segid in range(len(complete_phone_boundary_this_seg) - 1):
            frames = master_feats[pos + complete_phone_boundary_this_seg[segid]: pos + complete_phone_boundary_this_seg[segid+ 1]]
            float_frames = []
            for n in frames:
                float_frames.append(list(map(float, n))) # In Python 3, map() returns an iterable while, in Python 2, it returns a list. Any mathematical operation, like squaring (**), on a python iterable would throw similar error
#            print(float_frames) 
            if subsampling_flag <= 1 and subsampling_flag >= -1 : 
                master_seg_feats.append(list(np.average(np.array(float_frames), axis = 0)))
            elif subsampling_flag == 2:
                seg_repr_to_add =  list(np.append( np.average(np.array(float_frames[:int(len(float_frames)/2)]), axis = 0), np.average(np.array(float_frames[int(len(float_frames)/2):]), axis = 0) ))
                master_seg_feats.append(seg_repr_to_add)
            elif subsampling_flag == 3:
                seg_repr_to_add = list(np.concatenate( (np.average(np.array(float_frames[:int(len(float_frames)/3)]), axis = 0), np.average(np.array(float_frames[int(len(float_frames)/3):int(2 * len(float_frames)/3)]), axis = 0) , np.average(np.array(float_frames[int(2 * len(float_frames)/3):]), axis = 0)), axis = 0) )
                master_seg_feats.append(seg_repr_to_add)
            elif subsampling_flag > 3:
#                print("float_frames shape")
#                print(np.array(float_frames).shape)
                if len(float_frames) < subsampling_flag:
                    # we pad this too-short segment by 'edge' till the length of subsampling_flag
                    frames_to_pad = subsampling_flag - len(float_frames)
                    np_float_frames_pad = np.pad(np.array(float_frames), ((int(frames_to_pad/2) , frames_to_pad - int(frames_to_pad/2) ), (0,0)), 'edge')
#                    print(np_float_frames_pad)
#                    print("above:pad, below:original")
#                    print(np.array(float_frames))
#                    print("---")
                else:
                    np_float_frames_pad = np.array(float_frames)
#                seg_repr_to_add = list(    np.concatenate(         , axis = 0  )      )
#                print("dim of np_float_frames_pad:")
#                print(np_float_frames_pad.shape)
                seg_repr_to_add = np.average(np_float_frames_pad[:int(len(np_float_frames_pad)/subsampling_flag)], axis = 0)
                for index in range(1,subsampling_flag):
                    seg_repr_to_add = np.concatenate( (seg_repr_to_add, np.average(np_float_frames_pad[index * int(len(np_float_frames_pad)/subsampling_flag): (index+1) * int(len(np_float_frames_pad)/subsampling_flag)], axis = 0)), axis = 0  )
#                print("dim of seg_repr_to_add: \n")
#                print(seg_repr_to_add.shape)
                master_seg_feats.append(list(seg_repr_to_add)) 
            elif subsampling_flag == -3:
                seg_repr_to_add = list(np.concatenate( (np.average(np.array(float_frames), axis = 0),    np.average(np.array(float_frames[:int(len(float_frames)/2)]), axis = 0), np.average(np.array(float_frames[int(len(float_frames)/2):]), axis = 0) ), axis = 0))
                master_seg_feats.append(seg_repr_to_add)
            else:
                print("subsampling_flag error, check it's value ")
                exit(0)
        nsegments_per_utt[uttindex] = len(complete_phone_boundary_this_seg) - 1
#        if uttindex % 1000 == 0:
#            print(complete_phone_boundary_this_seg)
        pos = pos + nframes_per_utt[uttindex]
    print("master_seg_feats: %d by %d" % ( len(master_seg_feats), len(master_seg_feats[0]) ))
    print("nsegments_per_utt (first 10 utterances): " + str(nsegments_per_utt[0:10])) 
    print("Check: segment number: %d" % sum(nsegments_per_utt))
    X = np.array(master_seg_feats).astype(np.float64)
    if (subsampling_flag <= 1 and subsampling_flag >= -1 ) :
        subsampling_suffix = ''
    elif subsampling_flag == 2 or subsampling_flag == 3 or subsampling_flag == -3 or subsampling_flag > 3 :
        subsampling_suffix = "_subsamp" + str(subsampling_flag)
        print("Applying subsampling technique: subsampling_flag is %d" % (subsampling_flag ))
    if algorithm == 'hdbscan':
        print('hierarchical dbscan (HDBSCAN)')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1, core_dist_n_jobs=num_jobs) # The larger the value of min_samples you provide, the more conservative the clustering: more points will be declared as noise and clusters will be restricted to progressively more dense areas # min_cluster_sise: this is a relatively intuitive parameter to select:set it to the smallest size grouping that you wish to consider a cluster
#        print('X of size: %d by %d' % ( len(X), len(X[0]) ))
        clustering  = clusterer.fit(X)
        output_fullname = os.path.join(os.path.dirname(sys.argv[1]), 'output_' + os.path.basename(sys.argv[1]) + subsampling_suffix + 'hdbscan_seg_clusters' + str(nclusters)  +  '_rand_' + str(rand_state))
    elif algorithm == 'spectral':
        print('spectral clustering')
        clustering = SpectralClustering(n_clusters=nclusters, random_state=rand_state,  n_jobs = num_jobs, assign_labels = 'discretize' ).fit(X)
        #clustering = SpectralClustering(n_clusters=nclusters, random_state=rand_state,  n_jobs = num_jobs, assign_labels = 'kmeans' )]
        output_fullname = os.path.join(os.path.dirname(sys.argv[1]), 'output_' + os.path.basename(sys.argv[1]) + subsampling_suffix + 'spectral_seg_clusters' + str(nclusters)  +  '_rand_' + str(rand_state)) 
    elif algorithm == 'agglomerative':
        print('agglomerative clustering')
        clustering = AgglomerativeClustering(n_clusters=nclusters).fit(X) # compute_distances not found in 0.23.2 version
        #clustering = AgglomerativeClustering(n_clusters=nclusters,compute_distances=False).fit(X)
        output_fullname = os.path.join(os.path.dirname(sys.argv[1]), 'output_' + os.path.basename(sys.argv[1]) + subsampling_suffix + 'agglomerative_seg_clusters' + str(nclusters)  +  '_rand_' + str(rand_state))
    else:
        print('kmeans clustering')
        clustering = KMeans(n_clusters=nclusters, random_state=rand_state, verbose=1 , n_jobs = num_jobs).fit(X)
        output_fullname =  os.path.join(os.path.dirname(sys.argv[1]), 'output_' + os.path.basename(sys.argv[1]) + subsampling_suffix + '_seg_clusters' + str(nclusters)  +  '_rand_' + str(rand_state))
    cluster_labels = clustering.labels_
    print("Total number of clusters: %d" % ( cluster_labels.max() ))
    if algorithm == 'hdbscan':
        print("proportion noise samples by hdbscan: %f" % ( np.count_nonzero(cluster_labels == -1) / cluster_labels.size ))
    output_list_labels = list(cluster_labels)
    with open( output_fullname, 'w' ) as f_w:
        pos = 0
        for uttindex in range(len(uttid)):
            complete_phone_boundary_this_seg =  master_phone_boundary[uttindex] #str(nframes_per_utt[uttid])
            frame_label_this_utt = []
#            pos = 0
            #print("utt: %d, pos: %d" % ( uttindex, pos ))
            for n in range(len(complete_phone_boundary_this_seg) - 1):
                frame_label_this_utt.extend([ str(output_list_labels[pos + n]) ] * (complete_phone_boundary_this_seg[1+n]- complete_phone_boundary_this_seg[n]) )
            f_w.write(uttid[uttindex] + ' ' + ' '.join( frame_label_this_utt   ) + "\n" )
            pos += nsegments_per_utt[uttindex]  
            #print(complete_phone_boundary_this_seg)
# master_phone_boundary: every row corresponds to an utterance, like [ 5, 32, ..] meaning phone boundary index


print("%s: succeeded" % sys.argv[0] )
