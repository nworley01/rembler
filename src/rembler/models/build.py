from torch import nn


def build_model(model_type: str, in_channels: int, num_classes: int) -> nn.Module:
    if model_type == "small_cnn":
        from rembler.models.basic_cnn import SmallCNN

        return SmallCNN(in_channels=in_channels, num_classes=num_classes)
    elif model_type == "simple_dense":
        from rembler.models.basic_dense import SimpleDense

        return SimpleDense(
            in_channels=in_channels, num_classes=num_classes, sequence_length=25000
        )
    elif model_type == "cnn_bilstm":
        from rembler.models.cnn_bilstm import CNNBiLSTM

        return CNNBiLSTM(in_channels=in_channels, num_classes=num_classes)
    elif model_type == "implicit_cnn":
        from rembler.models.implicit_cnn import ImplicitFrequencyCNN

        return ImplicitFrequencyCNN(in_channels=in_channels, out_channels=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
