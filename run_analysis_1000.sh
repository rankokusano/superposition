#!/bin/bash
seed=0
restore_epoch=200
zeropadded_epoch=`printf %05d $restore_epoch`
test_batch_size=100

echo "Performing analysis on "$1" (1000-sequence variants)"

if [ $1 = "exp1" ]; then
    exp_config=exp1_l1_1000
    data_name=self_random_other_stay_1000
    visualize_data_indexes=(0 1 2 3 4)
    save_target_indexes="0 1 2 3 4"
elif [ $1 = "exp3" ]; then
    exp_config=exp3_1000
    data_name=self_random_other_stay_periodic_1000
    visualize_data_indexes=(0 1 2 3 4  100 101 102 103 104 200 201 202 203 204)
    save_target_indexes="0 1 2 3 4 100 101 102 103 104 200 201 202 203 204"
fi

rootdir="$(echo $PWD)"
resultdir=$rootdir/data/result/

cd $rootdir

python3 test.py \
    --exp_config $exp_config \
    --seed $seed \
    --cudnn_deterministic \
    --test_epoch $restore_epoch \
    --test_data_name $data_name \
    --test_batch_size $test_batch_size \
    --save_targets self_motion self_position other_motion other_position state \
    --data_load_memory

python3 test.py \
    --exp_config $exp_config \
    --seed $seed \
    --cudnn_deterministic \
    --test_epoch $restore_epoch \
    --test_data_name $data_name \
    --test_batch_size $test_batch_size \
    --save_targets vision \
    --data_load_memory \
    --test_modes test \
    --save_vision_img \
    --img_ext png \
    --save_targets_index $save_target_indexes

saveddir=$resultdir/$exp_config/$seed/test/$data_name/save
datadir=$rootdir/data/data/$data_name

cd $saveddir

ln -sf $rootdir/analyze ./
ln -sf $datadir/data.h5 ./

if [ $1 = "exp3" ]; then
    pca_exp_config=exp1_l1_1000
    pca_data_name=self_random_other_stay_1000
    pcadir=$resultdir/$pca_exp_config/$seed/test/$pca_data_name/save/$zeropadded_epoch/pca

    if [ ! -d "$pcadir" ]; then
        echo "ERROR: PCA directory not found: $pcadir"
        echo "Run './run_analysis_1000.sh exp1' first."
        exit 1
    fi

    mkdir -p ./$zeropadded_epoch
    cp -r $pcadir ./$zeropadded_epoch/
fi

for mode in eval test; do
    python3 analyze/plot_state.py \
        --epoch $restore_epoch \
        --mode $mode \
        --plot self_position \
        --plot_layer self

    python3 analyze/plot_state.py \
        --epoch $restore_epoch \
        --mode $mode \
        --plot other_position \
        --plot_layer other
done

mode=test

if [ $1 = "exp3" ]; then
    python3 analyze/plot_motion.py \
        --epoch $restore_epoch \
        --mode $mode
fi

for i in "${visualize_data_indexes[@]}"; do

    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run -a python3 analyze/plot_state.py \
        --epoch $restore_epoch \
        --mode $mode \
        --plot self_position \
        --plot_layer self \
        --animation \
        --idx $i

    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run -a python3 analyze/plot_state.py \
        --epoch $restore_epoch \
        --mode $mode \
        --plot other_position \
        --plot_layer other \
        --animation \
        --idx $i

    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run -a python3 analyze/record.py \
        --epoch $restore_epoch \
        --mode $mode \
        --idx $i
done

for dir in record/overview self_vision; do

    cd $saveddir/$zeropadded_epoch/$mode
    cd $dir

    ln -sf $rootdir/analyze/create_gif.py ./

    if [ $dir = "record/overview" ]; then
        gif_targets="input truth"
    elif [ $dir = "self_vision" ]; then
        gif_targets="input prediction truth"
    fi

    for i in "${visualize_data_indexes[@]}"; do
        python3 create_gif.py --idx $i --targets $gif_targets
    done
done

echo "Done."
