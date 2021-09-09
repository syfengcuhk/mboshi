This repo contains code used to implement models used in our INTERSPEECH 2021 paper:

S. Feng, P. \.Zelasko, L. Moro{-}Vel{\'{a}}zquez and O. Scharenborg, "Unsupervised Acoustic Unit Discovery by Leveraging a Language-independent Subword Discriminative Feature Representation", in Proc. INTERSPEECH 2021. Link: https://www.isca-speech.org/archive/pdfs/interspeech_2021/feng21_interspeech.pdf

Task:
Unsupervised acoustic unit discovery

Databases:
Evaluation: Mboshi, freely available for academic use.
Training: Globalphone 5 languages (Czech, French, Spanish, Mandarin and Thai), 8 Babel languages (Cantonese, Bengali, Vietnamese, Lao, Zulu, Amharic, Javanese and Georgian). Please note that GlobalPhone and IARPA BABEL Language resources are *NOT* freely available. Also note that the training of phone-level multilingual ASR systems with 5 or 13 GP & Babel languages is not included here, please refer to https://github.com/pzelasko/kaldi/tree/discophone/egs/discophone and paper: Feng et al. ICASSP 2021 "HOW PHONOTACTICS AFFECT MULTILINGUAL AND ZERO-SHOT ASR PERFORMANCE" for details of building phone-level (IPA) multilingual ASR systems.


Evaluation metrics:
(1) Normalized mutual information (NMI): a measure of clustering quality
(2) F-score: a measure of segmentation quality
Please refer to https://github.com/beer-asr/beer ([1]) for evaluation software.

See our paper for details of the approach.
S. Feng, P. \.Zelasko, L. Moro{-}Vel{\'{a}}zquez and O. Scharenborg, "Unsupervised Acoustic Unit Discovery by Leveraging a Language-Independent Subword Discriminative Feature Representation", in Proc. INTERSPEECH 2021.

References
[1] B. Yusuf, L. Ondel, L. Burget, J. Cernock ́y, and M. Saraclar, “Ahierarchical  subspace  model  for  language-attuned  acoustic  unitdiscovery,” CoRR, vol. abs/2011.03115, 2020. 
