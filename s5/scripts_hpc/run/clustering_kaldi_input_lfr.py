import sys
import os
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
import numpy as np

# input1 kaldi-format (text-format, not binary) feats.ark
# input2: int, num_jobs
# input3: int, random state
# input4: int, number of final clusters
# input5: int, frame subsampling factor
# performs k-means clustering by sklearn
# output frame-level cluster labels 
file_in = sys.argv[1]
num_jobs = int(sys.argv[2])
rand_state = int(sys.argv[3])
nclusters = int(sys.argv[4])
algorithm = sys.argv[5] # kmeans, spectral, dbscan
subsampling_factor = int(sys.argv[6])

#below for debugging
#num_jobs = 2
#rand_state = 0
#nclusters = 50
#algorithm = 'kmeans'
#subsampling_factor = 3

print("Subsampling factor : %d" % (subsampling_factor) )
print("Trying to read %s" % file_in)
print("Output file: %s" % os.path.join(os.path.dirname(sys.argv[1]), 'output_' + os.path.basename(sys.argv[1]) + '_clusters' + str(nclusters)  +  '_rand_' + str(rand_state)) + '_lfr' + str(subsampling_factor) )
with open(file_in, 'r') as f_in:
    uttid = []
    nframes_per_utt = []
    input_data = f_in.readlines()
    master_feats = []
    #utt_spec_feats = []
    print("Opend file %s, %d lines" % ( file_in, len(input_data) ))
    for this_line in input_data:
        this_line_splitted = this_line.strip().split()
        if this_line_splitted[-1] == '[':
            uttid.append(this_line_splitted[0])
            #nframes_this_utt = 0
            utt_spec_feats = []
        else:
            #nframes_this_utt += 1
            if this_line_splitted[-1] == ']' :
                #nframes_per_utt.append(nframes_this_utt)
                utt_spec_feats.append(this_line_splitted[:-1])
                master_feats.extend(utt_spec_feats[::subsampling_factor])
                nframes_per_utt.append(len(utt_spec_feats[::subsampling_factor]))
                utt_spec_feats = []
            else:
                utt_spec_feats.append(this_line_splitted)
print("In total %d frames, %d utterances" % ( len(master_feats), len(uttid) )) 
if len(uttid) != len(nframes_per_utt):
    print("uttid length != nframes_per_utt length, check input again")
else:
    print("Master feats: %d-by-%d matrix" % ( len(master_feats), len(master_feats[0]) ))     
    X = np.array(master_feats)
    # start estimating k-means
    if algorithm == 'spectral':
        print('spectral clustering') # still has Mem comsumption issue
        clustering = SpectralClustering(n_clusters=nclusters, random_state=rand_state,  n_jobs = num_jobs, assign_labels = 'discretize').fit(X)
        output_fullname = os.path.join(os.path.dirname(sys.argv[1]), 'output_' + os.path.basename(sys.argv[1]) + 'spec_clusters' + str(nclusters)  +  '_rand_' + str(rand_state) + '_lfr' + str(subsampling_factor)  ) 
    elif algorithm == 'dbscan':
        print('dbscan') # Still has bugs, output all '-1'
        clustering = DBSCAN(n_jobs = num_jobs, eps = 0.5).fit(X)
        output_fullname = os.path.join(os.path.dirname(sys.argv[1]), 'output_' + os.path.basename(sys.argv[1]) + 'dbscan_clusters' + str(nclusters)  +  '_rand_' + str(rand_state) + '_lfr' + str(subsampling_factor) ) 
    else:
        print('kmeans clustering')
        clustering = KMeans(n_clusters=nclusters, random_state=rand_state, verbose=1 , n_jobs = num_jobs).fit(X)
        output_fullname =  os.path.join(os.path.dirname(sys.argv[1]), 'output_' + os.path.basename(sys.argv[1]) + '_clusters' + str(nclusters)  +  '_rand_' + str(rand_state) + '_lfr' + str(subsampling_factor) )
    # start computing labels
    cluster_labels = clustering.labels_
    output_list_labels = list(cluster_labels) 
    with open( output_fullname, 'w' ) as f_w:
        # the first line
        f_w.write( uttid[0] + ' ' + " ".join( [ str(n) for n in output_list_labels[0:nframes_per_utt[0]]   for _ in range(subsampling_factor)  ]   )   + '\n')
        pos =  nframes_per_utt[0] 
        for index in range(1, len(uttid) ):
            output_this_line = uttid[index] + ' ' + " ".join([ str(n) for n in output_list_labels[ pos: pos +  nframes_per_utt[index]  ]  for _ in range(subsampling_factor)  ] )
            pos = pos + nframes_per_utt[index]
            f_w.write(output_this_line +  '\n'  )

print("%s: succeeded" % sys.argv[0] )

