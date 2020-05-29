"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 32

# params for source dataset
src_dataset = "CIFAR10"
# src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
# src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_encoder_restore = '../dynamic_sup/results/model/sup26_r26_bn_w4_resnet_ttt_r26_lr_0.1_bsz_128_aug_ttt/checkpoint_0200_0200.pth.tar'
src_classifier_restore = '../dynamic_sup/results/model/sup26_r26_bn_w4_resnet_ttt_r26_lr_0.1_bsz_128_aug_ttt/checkpoint_0200_0200.pth.tar'
src_model_trained = True

# params for target dataset
tgt_dataset = "CIFAR10-gaussian_noise,5"
# tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_encoder_restore = None
tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
d_input_dims = 256
d_hidden_dims = 500
d_output_dims = 2
# d_model_restore = "snapshots/ADDA-critic-final.pt"
d_model_restore = None

# params for training network
num_gpu = 1
num_epochs_pre = 0
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 1
log_step = 100
save_step = 100
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
