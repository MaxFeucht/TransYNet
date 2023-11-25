# Setup Settings
COLAB = False # whether to run on Google Colab
IMAGE_SIZE = 256  # the image size to the network
TRAIN_AUGMENT = False # whether to augment the training data
VAL_AUGMENT = False # whether to augment the validation data
AUGMENT_STRING = "_augment" if TRAIN_AUGMENT else ""
VAL_RATIO = 0.1 if TRAIN_AUGMENT else 0.1 # the ratio of the validation data used to determine best model

# Model Settings
MODEL_NAME = "transunet"  # unet, y_net_gen, y_net_gen_ffc, transunet, transynet
OUT_CHANNELS = 64 # the number of output channels for the first layer, is multiplied by up to 32 in TransYNet
BLOCK_NUM = 8 # the number of transformer blocks in TransUNet / TransYNet
MLP_DIM = 512 # the dimension of the MLP in the transformer blocks
DROPOUT_PROB = 0.15  # Dropout probability
SKIP_PROB = 0.05  # Stochastic depth rate

# Training Settings
BATCH_SIZE = 8 # if not TRAIN_AUGMENT else 32  # batch size
INIT_LR = 1e-3  # the initial learning rate
EPOCHS = 200  # the number of epochs to train for
LOAD = False  # Whether to load pretrained model
WEIGHTED = False  # weighted loss function
MARGIN_MASK = True # whether to use margin mask for training
DICE_OPTIMIZATION = False # whether to optimize based on DICE score
SCHEDULER_PATIENCE = 10 # the patience for the learning rate scheduler
SCHEDULER_FACTOR = 0.5 # the factor for the learning rate scheduler
