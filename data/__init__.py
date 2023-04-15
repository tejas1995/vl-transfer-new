from .memory_colors import MemoryColorsDataset
from .vicomte import VicomteDataset

COLOR_PROBE_TEMPLATES = [
                 "Q: What is the color of [DESCRIPTOR] [ITEM]? A: It is [MASK]. ",
                 "Q: What is the color of [DESCRIPTOR] [ITEM]? [SEP] A: It is [MASK]. ",
                 "Q: What is the colour of [DESCRIPTOR] [ITEM]? A: It is [MASK]. ",
                 "What is the color of [DESCRIPTOR] [ITEM]? [MASK]. ",
                 "What is the color of [DESCRIPTOR] [ITEM]? [SEP] [MASK]. ",
                 "What is the colour of [DESCRIPTOR] [ITEM]? [MASK]. ",
                 "The color of [ITEM] is [MASK]. ",
                 "The usual color of [DESCRIPTOR] [ITEM] is [MASK]. ",
                 "[DESCRIPTOR] [ITEM] usually has the color of [MASK]. ",
                 "What is the usual color of [DESCRIPTOR] [ITEM]? [MASK]. ",
                 "What is the usual color of [DESCRIPTOR] [ITEM]? [SEP] [MASK]. ",
                 "What is the typical color of [DESCRIPTOR] [ITEM]? [MASK]. ",
                 "What is the typical color of [DESCRIPTOR] [ITEM]? [SEP] [MASK]."
                ]

SHAPE_PROBE_TEMPLATES = ["Q: What is the shape of [DESCRIPTOR] [ITEM]? A: It is [MASK]. ",
                 "Q: What is the shape of [DESCRIPTOR] [ITEM]? [SEP] A: It is [MASK]. ",
                 "What is the shape of [DESCRIPTOR] [ITEM]? [MASK]. ",
                 "What is the shape of [DESCRIPTOR] [ITEM]? [SEP] [MASK]. ",
                 "The shape of [DESCRIPTOR] [ITEM] is [MASK]. ",
                 "The usual shape of [DESCRIPTOR] [ITEM] is [MASK]. ",
                 "[DESCRIPTOR] [ITEM] usually has the shape of [MASK]. ",
                 "What is the usual shape of [DESCRIPTOR] [ITEM]? [MASK]. ",
                 "What is the usual shape of [DESCRIPTOR] [ITEM]? [SEP] [MASK]. ",
                 "What is the typical shape of [DESCRIPTOR] [ITEM]? [MASK]. ",
                 "What is the typical shape of [DESCRIPTOR] [ITEM]? [SEP] [MASK]."
                ]

MATERIAL_PROBE_TEMPLATES = ["Q: What is the material of [DESCRIPTOR] [ITEM]? A: It is [MASK]. ",
                 "Q: What is the material of [DESCRIPTOR] [ITEM]? [SEP] A: It is [MASK]. ",
                 "What is the material of [DESCRIPTOR] [ITEM]? [MASK]. ",
                 "What is the material of [DESCRIPTOR] [ITEM]? [SEP] [MASK]. ",
                 "The material of [DESCRIPTOR] [ITEM] is [MASK]. ",
                 "The usual material of [DESCRIPTOR] [ITEM] is [MASK]. ",
                 "[DESCRIPTOR] [ITEM] usually has the material of [MASK]. ",
                 "What is the usual material of [DESCRIPTOR] [ITEM]? [MASK]. ",
                 "What is the usual material of [DESCRIPTOR] [ITEM]? [SEP] [MASK]. ",
                 "What is the typical material of [DESCRIPTOR] [ITEM]? [MASK]. ",
                 "What is the typical material of [DESCRIPTOR] [ITEM]? [SEP] [MASK]."
                ]

TEMPLATE_MAP = {
    'color': COLOR_PROBE_TEMPLATES,
    'shape': SHAPE_PROBE_TEMPLATES,
    'material': MATERIAL_PROBE_TEMPLATES,
}