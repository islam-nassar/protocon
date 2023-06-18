a=$1;
log='protocon_c10_40_456_run5_nomixup_weightedcpl_softpl_k250_inv_masking_proto_pl';
CUDA_VISIBLE_DEVICES=${a} \
python main_protocon.py \
    --num_classes 10 \
    --num_labels_per_class 4 \
    --x_u_split_file splits/c10_40_seed456.pkl \
    --split_random_seed 456 \
    -a wide_resnet28w2 \
    -j 16 \
    --use_amp \
    --strong_aug randaugment \
    --warmup_epochs 20 \
    --debias \
    --inverse_masking \
    --use_proto_pl \
    --refine \
    --weighted_cpl \
    --lambda-x 1 \
    --lambda-u 1 \
    --lambda-p-x 1 \
    --lambda-p-u 1 \
    --pseudo_label 'soft' \
    --lambda-c 1 \
    --consistency_loss 'instance' \
    --tau 0.95 \
    --hist_size 1 \
    --consistency_crit 'strict' \
    --mixup_alpha 8 \
    --mu 7 \
    --optimizer 'sgd' \
    --lr 0.03 \
    --batch-size 64 \
    --wd 5e-4 \
    --epochs 5001 \
    --proto-t 0.1 \
    --proto-k 250 \
    --proto-dual-lr 20 \
    --proto-ratio 0.9 \
    --proto-alpha 0.8 \
    --log ${log} \
    --dist-url 'tcp://localhost:'${RANDOM} --gpu 0 \
    /home/inas0003/data/external/cifar10_standard | tee log/${log}.log;