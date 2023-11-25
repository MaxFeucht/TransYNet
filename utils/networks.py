from nets.transunet import TransUNet
from nets.unet import UNet
from nets.ynet import YNet_general
from nets.transynet import TransYNet

def get_model(
        model_name, 
        img_dim = 256, 
        in_channels=3, 
        num_classes=4, 
        out_channels = 128, 
        head_num = 4, 
        mlp_dim = 512, 
        patch_dim = 16, 
        block_num = 8, 
        ratio=0.5,
        dropout_prob = 0.1,
        skip_prob = 0.1
        ):
    
    """Get model based on model name"""

    if model_name == "unet":

        model = UNet(
            in_channels = in_channels, 
            num_classes = num_classes,
            n_features = out_channels)
        
    elif model_name == "y_net_gen":

        model = YNet_general(
            in_channels = in_channels, 
            num_classes = num_classes, 
            init_features = out_channels,
            ffc=False)
        
    elif model_name == "y_net_gen_ffc":

        model = YNet_general(
            in_channels = in_channels, 
            num_classes = num_classes, 
            init_features = out_channels,
            ffc=True, 
            ratio_in=ratio)
        
    elif model_name == "transunet":

        model = TransUNet(
            img_dim = img_dim, 
            in_channels = in_channels, 
            out_channels = out_channels, 
            head_num = head_num, 
            mlp_dim = mlp_dim, 
            patch_dim = patch_dim, 
            block_num = block_num, 
            num_classes = num_classes,
            dropout_prob = dropout_prob,
            skip_prob = skip_prob
            )
        
    elif model_name == "transynet":

        model = TransYNet(
            img_dim = img_dim, 
            in_channels = in_channels, 
            out_channels = out_channels, 
            head_num = head_num, 
            mlp_dim = mlp_dim, 
            patch_dim = patch_dim, 
            block_num = block_num, 
            num_classes = num_classes,
            dropout_prob = dropout_prob,
            skip_prob = skip_prob
            )
    
        
    else:
        print(f"Model {model_name} name not found, please choose one of the following: unet, y_net_gen, y_net_gen_ffc, transunet, transynet")
        assert False

    return model