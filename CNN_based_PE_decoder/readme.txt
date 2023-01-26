해당 위치의 classifier_IEEE_2021_13_datasub_~~ .py 파일들은 모두 김동재 박사님께서 model based, model-free learning strategy 분류에 사용한 코드입니다.
최종적으로 사용하고 계신 것은 classifier_IEEE_2021_13_datasub_better_cnn4_16ch_tf2.py 이며, 나머지 파일은 혹시 참고가 될까 남겨 놓습니다. 
제가 위 코드를 참고하여 PE decoding을 한 파일은 다음과 같습니다. 
- pmb_decoder_mh2.py
: 기존의 2-stage MDT 환경에서 얻은 EEG data와, behavior fitting으로 얻은 SPE, RPE를 사용하는, SPE, RPE decoder 입니다. 

- pilot_decoder_mh3
: atari environment 환경에서 얻은 EEG data와, 임의로 레이블링 한 RPE를 사용하는, RPE decoder 입니다. 
(labeling 방식이 임의이므로 다른 방식을 찾는 것을 추천하며, 해당 방법이나 코드는 해당 폴더 내 labeling_RPE_pilot 내 참조…)

- pilot_decoder_mh_after_defense
: 기존에는 subject 2명에 대한 데이터를 한 번에 shuffling하여 사용하였는데, 디펜스 때 subject 1에 대해 training한 것을 subject 2에 대해 test, subject 2에 대해 training한 것을 subject 1에 대해 test 하는 것이 필요해 보인다는 코멘트를 들어서, 이에 대해 확인해보기 위한 파일.

**뒤에 _here 이 붙은 것은 로컬에서 돌리기 위한 파일이고 붙지 않은 것은 서버에서 돌리기 위한 것으로, 데이터를 부르는 디렉토리만 다르고 동일한 파일입니다. 

참고로 datamatching 엑셀 파일은, 김동재 박사님께서 주시고 가신 그 all_subj_struc (regressor파일 있는 것) 과 해당 data_sub와 순서가 달라 매칭한 것입니다. 
