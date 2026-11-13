import os
from importlib.resources import files

def get_asset_path(asset_name):
    """
    Get the full path to an asset file.
    
    Args:
        asset_name (str): Relative path from the assets directory
                         e.g., 'inspire_hand/left.urdf'
    
    Returns:
        Path: Full path to the asset file
    """
    assets_dir = files('dexmachina') / 'assets'
    return assets_dir / asset_name

def get_urdf_path(urdf_name):
    """Convenience function specifically for URDF files."""
    return get_asset_path(urdf_name)

def get_rl_config_path(cfg_name):
    # e.g. 'dexmachina/rl/configs/rl_games_ppo_cfg.yaml'
    assets_dir = files('dexmachina') / 'rl' / 'configs'
    asset_name = f'{cfg_name}.yaml'
    return assets_dir / asset_name
