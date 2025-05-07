import importlib.util
import torch
import os



def load_model_class(file_path: str, class_name: str):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def transfer_weights(old_model_path, new_model, device):
    """
    Transfers the weights from the first four layers of a pre-trained old model
    to the first four layers of a new model, without instantiating the old model.

    Args:
        old_model_path (str): Path to the .pth file containing the old model's weights.
        new_model (nn.Module): The new model to transfer the weights to.
    """
    def load_layer(new_layer, old_state_dict, layer_name):
        new_layer_state_dict = {}
        prefix = f"{layer_name}."
        for name, param in old_state_dict.items():
            if name.startswith(prefix):
                new_key = name[len(prefix):]
                if new_key in new_layer.state_dict():
                    new_layer_state_dict[new_key] = param
        new_layer.load_state_dict(new_layer_state_dict, strict=False)
    print("Loading old model weights from:", old_model_path)
    # Load the entire state_dict from the .pth file
    old_model_state_dict = torch.load(old_model_path, map_location=device)
    # Ensure the new model is in evaluation mode

    # Transfer weights layer by layer
    load_layer(new_model.bn_input, old_model_state_dict, "bn_input")
    load_layer(new_model.conv1, old_model_state_dict, "conv1")
    load_layer(new_model.conv2, old_model_state_dict, "conv2")
    load_layer(new_model.conv3, old_model_state_dict, "conv3")
    load_layer(new_model.conv4, old_model_state_dict, "conv4")
    print("Weights transferred successfully from old model to new model.")
    return new_model