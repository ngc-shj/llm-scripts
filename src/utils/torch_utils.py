import torch

def get_torch_dtype(dtype_str):
    """
    Convert a string representation of a dtype to the corresponding torch.dtype object.
    
    Args:
    dtype_str (str): String representation of the dtype.
    
    Returns:
    torch.dtype: The corresponding torch.dtype object.
    
    Raises:
    ValueError: If the input string doesn't correspond to a valid torch.dtype.
    """
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float,
        "float64": torch.float64,
        "double": torch.double,
        "float16": torch.float16,
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "short": torch.short,
        "int32": torch.int32,
        "int": torch.int,
        "int64": torch.int64,
        "long": torch.long,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    
    dtype = dtype_map.get(dtype_str.lower())
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype

if __name__ == "__main__":
    # Usage example
    config = {
        "torch_dtype": "bfloat16",
    }

    try:
        torch_dtype = get_torch_dtype(config["torch_dtype"])
        print(f"Converted dtype: {torch_dtype}")
    except ValueError as e:
        print(f"Error: {e}")

# Use the converted dtype
# model = Model(..., torch_dtype=torch_dtype)

