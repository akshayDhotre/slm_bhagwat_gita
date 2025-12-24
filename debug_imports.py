try:
    import sentence_transformers
    print("sentence_transformers imported successfully")
    print(f"Version: {sentence_transformers.__version__}")
except ImportError as e:
    print(f"Error importing sentence_transformers: {e}")

try:
    import transformers
    print("transformers imported successfully")
    print(f"Version: {transformers.__version__}")
except ImportError as e:
    print(f"Error importing transformers: {e}")

try:
    import torch
    print("torch imported successfully")
    print(f"Version: {torch.__version__}")
except ImportError as e:
    print(f"Error importing torch: {e}")
