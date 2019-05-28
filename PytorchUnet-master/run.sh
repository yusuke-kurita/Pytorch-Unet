#!/bin/bash

python=venv/bin/python

# # data preparation
musdb_root=/database/musicData/MUSDB18
wav_dir=wav
num_arg=24
wav_listname=${wav_dir}
# ${python} prepare_data.py \
#     --src_dir ${musdb_root} \
#     --dst_dir ${wav_dir} \
#     --num_arg ${num_arg}

# training wav data
fd wav ${wav_dir}/train | sort > "${wav_listname}.csv"
# make dataset
wav_listname=wav
mag_spec_root_dir=mag_spec
fs=44100
frame_size=4096
shift_size=2048
batch_length=512
num_worker=1
mag_spec_listname=fs${fs}_frame${frame_size}_shift${shift_size}_batch${batch_length}
mag_spec_dir=${mag_spec_root_dir}/${mag_spec_listname}
# ${python} make_dataset.py \
#     --src_file "${wav_listname}.csv" \
#     --dst_dir "${mag_spec_dir}" \
#     --fs ${fs} \
#     --frame_size ${frame_size} \
#     --shift_size ${shift_size} \
#     --num_worker ${num_worker}

# training dataset path
fd npy "${mag_spec_dir}" | sort  > "${mag_spec_listname}.csv"
# calc satics
stat_dir=stat
stat_listname=${stat_dir}/${mag_spec_listname}
# ${python} calc_stats.py \
#     --src_file "${mag_spec_listname}.csv" \
#     --dst_dir ${stat_dir}

# train network
model_dir=model
batch_size=64
lr=1e-4
seed=0
num_epoch=200
num_interval=20
ratio=0.8
${python} train.py \
    --src_file "${mag_spec_listname}.csv" \
    --stats_file "${stat_listname}.npy" \
    --dst_dir ${model_dir} \
    --batch_length ${batch_length} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --seed ${seed} \
    --num_epoch ${num_epoch} \
    --num_worker ${num_worker} \
    --num_interval ${num_interval} \
    --ratio ${ratio}
