import sys

# This script converts Mboshi reference phone alignments to phoneme alignments by a lookup table created by Siyuan Feng.
# Use: python3 phone2phoneme_conversion.py $ref_phone_ali $phone2phoneme_table $output_ref_phoneme_ali

# if you run scripts in  BEER/beer/recipes/hshmm/, a reference phone alignment file of Mboshi should be located in data/mboshi/full/ali
# The phone2phoneme_table is located at ../../references/phone2phoneme_sym_SF_draft.txt
 

file_phone_ali = sys.argv[1]
phone2phoneme = sys.argv[2]
file_phoneme_ali = sys.argv[3]

print("Convert phone alignment %s to phoneme alignment %s " % (file_phone_ali, file_phoneme_ali))
print("Based on the phone2phoneme mapping %s" % phone2phoneme)
with open(phone2phoneme, 'r') as f_mapping:
	mapping = f_mapping.readlines()
	num_phones = len(mapping)
	phonemeid_of_phone = [0] * num_phones
	for index in range(num_phones):
			phonemeid_of_phone[int(mapping[index].strip().split(' ')[0]) - 1] = int(mapping[index].strip().split(' ')[1])
	with open(file_phone_ali, 'r') as f_phone_ali:
			content = f_phone_ali.readlines()
			num_utts = len(content)
			with open(file_phoneme_ali, 'w') as f_phoneme_ali_w:
					for utt_index in range(num_utts):
							this_utt_old = content[utt_index].strip().split(' ')
							f_phoneme_ali_w.write( this_utt_old[0] + " ") # utterance name
							for token_index in range(1,len(this_utt_old)):
									# Print phoneme one by one
									if this_utt_old[token_index].startswith('phn'):
											# A regular phone
											f_phoneme_ali_w.write( 'phoneme' + str(phonemeid_of_phone[int(this_utt_old[token_index][3:]) - 1]) + ' ')  
									else:
											# silence or <unk>, thus don't convert
											f_phoneme_ali_w.write(this_utt_old[token_index] + ' ')
							# End of an utterance.
							f_phoneme_ali_w.write('\n')




