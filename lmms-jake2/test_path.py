try:
    from llava.mm_utils import get_model_name_from_path
except ImportError as e:
    print("ImportError:", e)