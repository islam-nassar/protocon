model=$1;
python eval_lincls.py \
    -a wide_resnet28w2 \
    --lr 0.3 \
    --pretrained ${model} \
    --dist-url 'tcp://localhost:'${RANDOM} --multiprocessing-distributed --world-size 1 --rank 0 \
    /home/inas0003/data/external/cifar10_standard
