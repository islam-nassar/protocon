a=$1;
log='protocon_c10_40_456_sanity_capture_stats_noLc';
CUDA_VISIBLE_DEVICES=${a} \
python main_protocon.py \
    --num_classes 10 \
    --proto-num-ins 50000 \
    --num_labels_per_class 4 \
    --x_u_split_file splits/c10_40_seed456.pkl \
    --split_random_seed 456 \
    --capture_stats \
    -a wide_resnet28w2 \
    -j 16 \
    --strong_aug randaugment \
    --warmup_epochs 10000 \
    --lambda-x 1 \
    --lambda-u 1 \
    --lambda-p 0 \
    --lambda-c 0 \
    --tau_warmup 0.95 \
    --tau 0.95 \
    --hist_size 3 \
    --mixup_alpha 8 \
    --mu 7 \
    --debias \
    --optimizer sgd \
    --lr 0.03 \
    --batch-size 64 \
    --wd 5e-4 \
    --epochs 5001 \
    --proto-t 0.1 \
    --proto-num-head 1 \
    --proto-k 100 \
    --proto-dual-lr 20 \
    --proto-ratio 0.9 \
    --proto-alpha 0.2 \
    --log ${log} \
    --dist-url 'tcp://localhost:'${RANDOM} --gpu 0 \
    /home/inas0003/data/external/cifar10_standard | tee log/${log}.log;