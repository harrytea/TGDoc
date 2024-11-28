from .clip_encoder import CLIPVisionTower
from .eva_encoder import EVACILPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("eva_clip_g"):
        vision_model = EVACILPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        return vision_model
    raise ValueError(f'Unknown vision tower: {vision_tower}')
