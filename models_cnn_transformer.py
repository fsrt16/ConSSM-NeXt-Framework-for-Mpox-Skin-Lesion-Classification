import timm

# Full dictionary from your original code
RECOMMENDED_IMG_SIZES = {
    "resnet18": 224, "resnet34": 224, "resnet50": 224, "resnet101": 224, "resnext50_32x4d": 224,
    "wide_resnet50_2": 224, "vgg11_bn": 224, "vgg13_bn": 224, "vgg16_bn": 224, "vgg19_bn": 224,
    "densenet121": 224, "densenet169": 224, "densenet201": 224, "efficientnet_b0": 224,
    "efficientnet_b1": 240, "efficientnet_b2": 260, "mobilenetv2_100": 224, "mobilenetv3_large_100": 224,
    "inception_v3": 299, "xception": 299, "vit_base_patch16_224": 224, "vit_small_patch16_224": 224,
    "vit_tiny_patch16_224": 224, "swin_base_patch4_window7_224": 224, "swin_small_patch4_window7_224": 224,
    "swin_tiny_patch4_window7_224": 224, "deit_base_distilled_patch16_224": 224, "deit_small_distilled_patch16_224": 224,
    "maxvit_tiny_rw_224": 224, "convnext_tiny": 224
}

# The full list of 20 CNNs
ALL_20_CNN_MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnext50_32x4d", "wide_resnet50_2",
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn", "densenet121", "densenet169", "densenet201",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "mobilenetv2_100",
    "mobilenetv3_large_100", "inception_v3", "xception",
]

# The full list of 10 Transformers
ALL_10_TRANSFORMER_MODELS = [
    "vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224",
    "swin_base_patch4_window7_224", "swin_small_patch4_window7_224", "swin_tiny_patch4_window7_224",
    "deit_base_distilled_patch16_224", "deit_small_distilled_patch16_224",
    "maxvit_tiny_rw_224", "convnext_tiny"
]

def get_model(model_name, num_classes, pretrained=True):
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        # Handle Inception auxiliary logits which can crash standard loops
        if "inception" in model_name and hasattr(model, 'aux_logits'):
            model.aux_logits = False
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None