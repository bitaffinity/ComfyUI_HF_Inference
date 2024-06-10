try:
    import comfy.utils
except ImportError:
    pass
else:
    NODE_CLASS_MAPPINGS = {}

    from .Text.nodes import NODE_CLASS_MAPPINGS as Text_Nodes
    NODE_CLASS_MAPPINGS.update(Text_Nodes)

    from .Image.nodes import NODE_CLASS_MAPPINGS as Image_Nodes
    NODE_CLASS_MAPPINGS.update(Image_Nodes)

    NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}

    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']