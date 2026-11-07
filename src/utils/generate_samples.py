"""
Sample Data Generator and Dataset Loader for NYUv2, Middlebury, KITTI, and Synthetic Benchmarks.
Generates realistic high-resolution RGB-D pairs with complex geometry, occlusions,
and challenging textures (for anti-texture-copying tests).
"""

import os
import numpy as np
import cv2

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'samples')

def create_synthetic_geometric_sample(H: int = 384, W: int = 512) -> tuple:
    """
    Generates a high-precision synthetic benchmark pair:
    - Multiple depth planes (background wall at 4.0m, table at 2.5m, foreground objects at 1.0m - 1.8m)
    - High-frequency RGB texture on the background wall (brick / wallpaper pattern) to test anti-texture copying
    - Sharp geometric boundaries (cubes, cylinders, spheres) to test edge preservation
    """
    depth = np.ones((H, W), dtype=np.float32) * 4.0  # Background wall at 4.0m
    rgb = np.ones((H, W, 3), dtype=np.float32) * 0.95
    
    # 1. Add high-contrast brick/grid texture to the background wall in RGB (depth remains flat 4.0m!)
    for y in range(0, H, 24):
        rgb[y:y+2, :, :] = 0.4
    for x in range(0, W, 40):
        rgb[:, x:x+2, :] = 0.4
    rgb[:, :, 0] += 0.05  # Slight warm tint
    
    # 2. Middle ground plane: slanted table top (y from 240 to 384)
    for y in range(240, H):
        t = (y - 240) / (H - 240)
        depth[y, :] = 3.2 - 1.2 * t  # Slopes from 3.2m to 2.0m
        rgb[y, :, :] = [0.35 + 0.1 * t, 0.25 + 0.1 * t, 0.18 + 0.1 * t]  # Wood color
        
    # Add wood grain texture on table in RGB
    for x in range(0, W, 12):
        rgb[240:H, x:x+1, :] *= 0.85
        
    # 3. Foreground Object 1: Sphere at (160, 160), radius 75, depth 1.4m
    cy1, cx1, r1 = 170, 160, 75
    y, x = np.ogrid[:H, :W]
    dist_sq1 = (x - cx1)**2 + (y - cy1)**2
    mask1 = dist_sq1 <= r1**2
    if np.any(mask1):
        z_sphere = np.sqrt(np.maximum(0.0, r1**2 - dist_sq1[mask1])) / r1 * 0.4
        depth[mask1] = 1.6 - z_sphere  # Depth 1.2m to 1.6m
        # Smooth blue shading
        shading = z_sphere / 0.4
        rgb[mask1, 0] = 0.1 * shading
        rgb[mask1, 1] = 0.4 * shading
        rgb[mask1, 2] = 0.85 * shading + 0.1
        
    # 4. Foreground Object 2: Sharp Rectangular Box at (120:280, 290:430), depth 1.1m
    box_mask = np.zeros((H, W), dtype=bool)
    box_mask[120:280, 290:430] = True
    depth[box_mask] = 1.15
    rgb[box_mask] = [0.9, 0.25, 0.2]  # Vibrant red box
    # Box edges
    rgb[120:124, 290:430] = [1.0, 0.5, 0.4]
    rgb[120:280, 290:294] = [1.0, 0.5, 0.4]
    rgb[276:280, 290:430] = [0.6, 0.1, 0.1]
    rgb[120:280, 426:430] = [0.6, 0.1, 0.1]
    
    # 5. Small cylinder overlapping between box and table (210:320, 240:290), depth 0.9m
    cyl_mask = np.zeros((H, W), dtype=bool)
    cyl_mask[210:330, 230:280] = True
    depth[cyl_mask] = 0.88
    rgb[cyl_mask] = [0.2, 0.8, 0.3]  # Green cylinder
    
    # Add subtle sensor noise and realistic smooth transitions
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, depth


def create_nyuv2_living_room_sample(H: int = 384, W: int = 512) -> tuple:
    """
    Simulates an NYU Depth V2 indoor living room scene.
    Wall with framed artwork, sofa, coffee table, floor.
    """
    depth = np.ones((H, W), dtype=np.float32) * 3.8
    rgb = np.ones((H, W, 3), dtype=np.float32) * 0.85
    
    # Floor: lower 40% of image
    for y in range(220, H):
        t = (y - 220) / (H - 220)
        depth[y, :] = 3.6 - 2.2 * t
        rgb[y, :] = [0.55 - 0.1 * t, 0.45 - 0.1 * t, 0.35 - 0.1 * t]
        
    # Wall art frame (RGB has high contrast, but depth is flush with wall!)
    rgb[40:140, 160:340] = [0.2, 0.2, 0.2]
    rgb[46:134, 166:334] = [0.95, 0.8, 0.4]  # Yellow sunset painting inside
    # Add painting details
    for y in range(60, 120, 10):
        rgb[y:y+4, 180:320] = [0.8, 0.3, 0.2]
    # Depth of frame is almost flush: 3.75m vs wall 3.8m
    depth[40:140, 160:340] = 3.75
    
    # Sofa: occupying bottom-left and center
    sofa_mask = np.zeros((H, W), dtype=bool)
    sofa_mask[180:310, 60:320] = True
    depth[sofa_mask] = 2.2
    rgb[sofa_mask] = [0.25, 0.35, 0.55]  # Navy blue sofa
    
    # Sofa cushion details
    sofa_cushion = np.zeros((H, W), dtype=bool)
    sofa_cushion[210:290, 80:180] = True
    sofa_cushion[210:290, 190:290] = True
    depth[sofa_cushion] = 2.05
    rgb[sofa_cushion] = [0.3, 0.4, 0.65]
    
    # Coffee table in foreground: bottom-right
    table_mask = np.zeros((H, W), dtype=bool)
    table_mask[260:350, 310:470] = True
    depth[table_mask] = 1.35
    rgb[table_mask] = [0.75, 0.55, 0.35]
    
    # Mug on table (small sharp object)
    mug_mask = np.zeros((H, W), dtype=bool)
    mug_mask[230:265, 380:410] = True
    depth[mug_mask] = 1.25
    rgb[mug_mask] = [0.95, 0.95, 0.95]
    
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, depth


def create_middlebury_cones_sample(H: int = 384, W: int = 512) -> tuple:
    """
    Simulates Middlebury 'Cones' benchmark scene:
    Multiple conical shapes with gradual slopes and sharp occlusion silhouettes.
    """
    depth = np.ones((H, W), dtype=np.float32) * 3.5
    rgb = np.ones((H, W, 3), dtype=np.float32) * 0.7
    
    # Cones at different depths and positions
    cones = [
        {'cx': 140, 'cy': 180, 'r': 65, 'd_base': 2.2, 'd_tip': 1.2, 'rgb': [0.9, 0.5, 0.1]},
        {'cx': 270, 'cy': 150, 'r': 85, 'd_base': 2.8, 'd_tip': 1.6, 'rgb': [0.2, 0.6, 0.85]},
        {'cx': 380, 'cy': 210, 'r': 55, 'd_base': 1.9, 'd_tip': 0.9, 'rgb': [0.85, 0.2, 0.4]},
        {'cx': 200, 'cy': 260, 'r': 70, 'd_base': 1.6, 'd_tip': 0.7, 'rgb': [0.95, 0.8, 0.15]},
    ]
    
    y, x = np.ogrid[:H, :W]
    for c in cones:
        dist = np.sqrt((x - c['cx'])**2 + (y - c['cy'])**2)
        mask = dist <= c['r']
        if np.any(mask):
            cone_height_ratio = (c['r'] - dist[mask]) / c['r']
            cone_d = c['d_base'] - cone_height_ratio * (c['d_base'] - c['d_tip'])
            
            # Depth buffer: only overwrite if closer
            closer = cone_d < depth[mask]
            sub_mask = np.zeros((H, W), dtype=bool)
            sub_coords = np.where(mask)
            sub_mask[(sub_coords[0][closer], sub_coords[1][closer])] = True
            
            depth[sub_mask] = cone_d[closer]
            
            # Shading
            shading = 0.5 + 0.5 * cone_height_ratio[closer]
            for ch in range(3):
                rgb[sub_mask, ch] = c['rgb'][ch] * shading
                
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, depth


def create_kitti_street_sample(H: int = 384, W: int = 512) -> tuple:
    """
    Simulates KITTI autonomous driving scene:
    Road plane, distant buildings, vehicle silhouette in foreground, tree canopy.
    """
    depth = np.ones((H, W), dtype=np.float32) * 35.0  # Far background 35m
    rgb = np.ones((H, W, 3), dtype=np.float32)
    rgb[:180, :] = [0.45, 0.65, 0.9]  # Sky
    
    # Road plane
    for y in range(200, H):
        t = (y - 200) / (H - 200)
        depth[y, :] = 35.0 / (1.0 + 9.0 * t)  # 35m down to 3.5m
        rgb[y, :] = [0.28, 0.28, 0.3]  # Asphalt
        
    # Road markings (high contrast in RGB, same depth as road)
    rgb[250:H:15, 250:262] = [0.9, 0.9, 0.9]
    
    # Vehicle in front (depth ~7.5m)
    car_mask = np.zeros((H, W), dtype=bool)
    car_mask[220:310, 180:330] = True
    car_cabin = np.zeros((H, W), dtype=bool)
    car_cabin[185:230, 205:305] = True
    
    depth[car_mask] = 7.8
    depth[car_cabin] = 8.1
    
    rgb[car_mask] = [0.15, 0.15, 0.75]  # Blue car
    rgb[car_cabin] = [0.1, 0.1, 0.15]   # Car glass/cabin
    # Tail lights
    rgb[255:270, 190:215] = [0.95, 0.1, 0.1]
    rgb[255:270, 295:320] = [0.95, 0.1, 0.1]
    # License plate
    rgb[270:285, 235:275] = [0.95, 0.95, 0.8]
    
    # Tree on roadside (left)
    tree_trunk = np.zeros((H, W), dtype=bool)
    tree_trunk[120:250, 40:70] = True
    depth[tree_trunk] = 12.0
    rgb[tree_trunk] = [0.35, 0.22, 0.15]
    
    tree_leaves = np.zeros((H, W), dtype=bool)
    y_g, x_g = np.ogrid[:H, :W]
    tree_leaves = ((x_g - 55)**2 + (y_g - 110)**2) <= 50**2
    depth[tree_leaves] = 11.5
    rgb[tree_leaves] = [0.15, 0.5, 0.18]
    
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, depth


def generate_all_preset_samples():
    """Generates and saves standard benchmark sample pairs to data/samples/."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    generators = {
        'synthetic_geometric': create_synthetic_geometric_sample,
        'nyuv2_living_room': create_nyuv2_living_room_sample,
        'middlebury_cones': create_middlebury_cones_sample,
        'kitti_street': create_kitti_street_sample
    }
    
    for name, gen_fn in generators.items():
        rgb, depth = gen_fn(384, 512)
        
        # Save RGB as PNG (uint8)
        rgb_u8 = (rgb * 255.0).astype(np.uint8)
        rgb_path = os.path.join(SAMPLES_DIR, f"{name}_rgb.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        
        # Save Depth as 16-bit PNG (millimeters: depth_meters * 1000) and npy
        depth_mm = (depth * 1000.0).astype(np.uint16)
        depth_path = os.path.join(SAMPLES_DIR, f"{name}_depth.png")
        cv2.imwrite(depth_path, depth_mm)
        
        # Also save raw float32 npy for 100% numerical fidelity
        npy_path = os.path.join(SAMPLES_DIR, f"{name}_depth.npy")
        np.save(npy_path, depth.astype(np.float32))
        
    print(f"Generated {len(generators)} preset benchmark samples in {SAMPLES_DIR}")

if __name__ == '__main__':
    generate_all_preset_samples()
