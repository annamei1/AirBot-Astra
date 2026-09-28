"""
Position Calculators for Brick Detection

Provides geometric calculations for:
- Head camera: Convert depth + mask to 3D position + area
- Hand-eye camera: Fine positioning with known reference
- Push planning: Calculate push candidates for toppling bricks
- Yaw refinement: Correct yaw using top surface detection
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List


# ==================== Configuration ====================
# Relative to this file, not an absolute path into another checkout. An absolute path here
# silently makes this tree read another tree's calibration, which is the hardest kind of
# coupling to notice: everything runs, the numbers are just someone else's.
CALIB_DIR = Path(__file__).resolve().parent.parent / "calibration"

HEAD_INTRINSICS = {
    'fx': 607.15, 
    'fy': 607.02, 
    'cx': 324.25, 
    'cy': 248.46
}

# ==================== Brick Geometry ====================
BRICK_L = 0.11   # 11cm - longest dimension
BRICK_W = 0.05   # 5cm - medium dimension
BRICK_H = 0.025  # 2.5cm - shortest dimension

# ==================== Push Planning Constants ====================
PUSH_DISTANCE_MIN = 0.07
PUSH_DISTANCE_RECOMMENDED = 0.09
PUSH_APPROACH_OFFSET = 0.06
PUSH_START_OFFSET = 0.03
PUSH_CONTACT_HEIGHT_RATIO = 0.70
MIN_CONTACT_HEIGHT_ABOVE_GROUND = 0.025
SAFE_APPROACH_HEIGHT = 0.10


# ==================== Dynamic Z Compensator ====================

class DynamicZCompensator:
    """Dynamic Z-axis Compensator"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.model_type = "linear"
        self.coefficients = {
            'intercept': 0.0,
            'x_coeff': 0.0,
            'y_coeff': 0.0,
            'xy_coeff': 0.0,
            'x2_coeff': 0.0,
            'y2_coeff': 0.0,
        }
        self.enabled = False
        self.constant_z_offset = 0.0
        self._load_config()
    
    def _load_config(self):
        if not self.config_path.exists():
            print(f"[Z-Comp] Config file not found: {self.config_path}")
            return
        
        try:
            with open(self.config_path) as f:
                data = json.load(f)
            
            dz = data.get('dynamic_z_compensation', {})
            self.enabled = dz.get('enabled', False)
            self.model_type = dz.get('model', 'linear')
            self.coefficients = dz.get('coefficients', self.coefficients)
            self.constant_z_offset = dz.get('constant_z_offset', 0.0)
            
            if self.enabled:
                print(f"[Z-Comp] Loaded dynamic compensation model: {self.model_type}")
                if abs(self.constant_z_offset) > 1e-6:
                    print(f"[Z-Comp] Extra Z offset: {self.constant_z_offset*1000:+.1f} mm")
        except Exception as e:
            print(f"[Z-Comp] Failed to load config: {e}")
    
    def compute_compensation(self, x: float, y: float) -> float:
        if not self.enabled:
            return self.constant_z_offset  # 即使模型禁用，也应用额外偏移
        
        c = self.coefficients
        
        if self.model_type == "linear":
            base = c.get('intercept', 0.0) + c.get('x_coeff', 0.0) * x + c.get('y_coeff', 0.0) * y
        elif self.model_type == "quadratic":
            base = (c.get('intercept', 0.0) + 
                    c.get('x_coeff', 0.0) * x + 
                    c.get('y_coeff', 0.0) * y +
                    c.get('xy_coeff', 0.0) * x * y +
                    c.get('x2_coeff', 0.0) * x * x +
                    c.get('y2_coeff', 0.0) * y * y)
        else:
            base = 0.0
        
        return base + self.constant_z_offset


# ==================== Yaw Refinement ====================

def refine_upright_brick_yaw(
    mask: np.ndarray,
    depth: np.ndarray,
    original_yaw: float,
    brick_pose: str = "upright",
    brick_size_LWH: Optional[List[float]] = None,
    depth_threshold_mm: float = 15.0,
    min_aspect_ratio: float = 1.3,
    tf_matrix: Optional[np.ndarray] = None,
) -> Tuple[float, bool]:
    """
    Refine yaw for upright/side bricks using depth-based top surface detection.
    
    The refined yaw is normalized to [0, π] range (0° to 180°) for push direction.
    This ensures consistent push direction calculation.
    
    For UPRIGHT bricks:
    - Top surface is W×H (5cm × 2.5cm)
    - Long edge = W axis
    
    For SIDE bricks:
    - Top surface is L×H (11cm × 2.5cm)  
    - Long edge = L axis
    
    Args:
        mask: Binary mask (H, W) or (1, H, W)
        depth: Depth image in mm
        original_yaw: Original yaw from standard estimation (radians) - in BASE FRAME
        brick_pose: "upright" or "side"
        brick_size_LWH: Brick dimensions [L, W, H]
        depth_threshold_mm: Depth threshold for top surface extraction
        min_aspect_ratio: Minimum long/short ratio to trust the result
        tf_matrix: Camera to base_link transform matrix (4x4)
        
    Returns:
        Tuple of (refined_yaw, was_refined)
        - refined_yaw: Yaw in [0, π] range (0° to 180°) in BASE FRAME
        - was_refined: True if refinement succeeded
    """
    from perception.sam3_segmenter import extract_top_surface_yaw
    
    if brick_size_LWH is None:
        brick_size_LWH = [BRICK_L, BRICK_W, BRICK_H]
    
    if brick_pose not in ["upright", "side"]:
        return original_yaw, False
    
    # Extract yaw from top surface (in CAMERA frame)
    yaw_cam, info = extract_top_surface_yaw(
        mask, depth, depth_threshold_mm, min_aspect_ratio
    )
    
    if yaw_cam is None:
        print(f"    [YAW REFINE] Failed: {info.get('error', 'unknown')}")
        return original_yaw, False
    
    # Convert from camera frame to base frame
    if tf_matrix is not None:
        yaw_base = yaw_cam + np.arctan2(tf_matrix[1, 0], tf_matrix[0, 0]) + np.pi
    else:
        print(f"    [YAW REFINE] Warning: No TF matrix provided")
        yaw_base = yaw_cam
    
    # ===== Normalize to [0, π] range (0° to 180°) =====
    # This ensures consistent push direction calculation
    while yaw_base < 0:
        yaw_base += np.pi
    while yaw_base >= np.pi:
        yaw_base -= np.pi
    
    refined_yaw = float(yaw_base)
    
    # Also normalize original_yaw to [0, π] for comparison
    orig_normalized = original_yaw
    while orig_normalized < 0:
        orig_normalized += np.pi
    while orig_normalized >= np.pi:
        orig_normalized -= np.pi
    
    # Calculate angular difference
    yaw_diff = abs(refined_yaw - orig_normalized)
    if yaw_diff > np.pi / 2:
        yaw_diff = np.pi - yaw_diff
    
    # Debug output
    edge_type = "W" if brick_pose == "upright" else "L"
    rect_info = info.get('rect_info', {})
    
    print(f"    [YAW REFINE] Top surface: {rect_info.get('long_edge_px', 0):.1f}×{rect_info.get('short_edge_px', 0):.1f}px, "
          f"ratio={info.get('aspect_ratio', 0):.2f}, {edge_type}-axis(cam)={info.get('long_axis_angle_deg', 0):.1f}°")
    print(f"    [YAW REFINE] Original: {np.degrees(original_yaw):.1f}° → Normalized: {np.degrees(orig_normalized):.1f}°")
    print(f"    [YAW REFINE] Refined:  {np.degrees(refined_yaw):.1f}° (diff: {np.degrees(yaw_diff):.1f}°)")
    
    # Always use refined yaw for upright/side bricks
    # was_refined=True indicates top surface detection succeeded
    return refined_yaw, True

# ==================== Area Calculation ====================

def calculate_real_area(
    mask: np.ndarray, 
    depth: np.ndarray, 
    fx: float, 
    fy: float
) -> Tuple[float, int]:
    """
    Calculate real-world area of a segmented region.
    
    Args:
        mask: binary mask (H, W) or boolean mask
        depth: depth image in mm
        fx, fy: camera focal lengths
        
    Returns:
        Tuple of (area_cm2, area_pixels)
    """
    mask_bool = mask > 0.5 if mask.dtype != bool else mask
    pixel_count = int(np.sum(mask_bool))
    
    if pixel_count == 0:
        return 0.0, 0
    
    valid_depth = depth[mask_bool]
    valid_depth = valid_depth[(valid_depth > 100) & (valid_depth < 2000)]
    
    if len(valid_depth) == 0:
        return 0.0, pixel_count
    
    median_depth_m = np.median(valid_depth) / 1000.0
    pixel_size_m2 = (median_depth_m / fx) * (median_depth_m / fy)
    area_m2 = pixel_count * pixel_size_m2
    area_cm2 = area_m2 * 10000
    
    return area_cm2, pixel_count


# ==================== OBB Collision Detection ====================

def get_obb_corners(cx: float, cy: float, yaw: float, 
                    half_L: float, half_W: float) -> np.ndarray:
    """
    Get the 4 corners of an oriented bounding box.
    
    Args:
        cx, cy: Center position
        yaw: Rotation angle (radians), L-axis direction
        half_L, half_W: Half dimensions
        
    Returns:
        np.ndarray of shape (4, 2) with corner coordinates
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # Local coordinates of corners (L along yaw direction, W perpendicular)
    local_corners = np.array([
        [+half_L, +half_W],
        [+half_L, -half_W],
        [-half_L, -half_W],
        [-half_L, +half_W],
    ])
    
    # Rotation matrix
    R = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw],
    ])
    
    # Transform to world coordinates
    world_corners = (R @ local_corners.T).T + np.array([cx, cy])
    
    return world_corners


def check_obb_collision(
    cx1: float, cy1: float, yaw1: float, half_L1: float, half_W1: float,
    cx2: float, cy2: float, yaw2: float, half_L2: float, half_W2: float,
    margin: float = 0.01,
) -> bool:
    """
    Check if two oriented bounding boxes collide using Separating Axis Theorem (SAT).
    
    Args:
        cx1, cy1: Center of first OBB
        yaw1: Rotation of first OBB (radians)
        half_L1, half_W1: Half-dimensions of first OBB
        cx2, cy2, yaw2, half_L2, half_W2: Same for second OBB
        margin: Safety margin added to first OBB (meters)
        
    Returns:
        True if collision detected
    """
    # Get corners of both boxes (add margin to first box)
    corners1 = get_obb_corners(cx1, cy1, yaw1, half_L1 + margin, half_W1 + margin)
    corners2 = get_obb_corners(cx2, cy2, yaw2, half_L2, half_W2)
    
    def get_axes(corners: np.ndarray) -> List[np.ndarray]:
        """Get the 2 unique edge normals (axes) for SAT test."""
        axes = []
        for i in range(2):  # Only need 2 axes per rectangle
            edge = corners[(i + 1) % 4] - corners[i]
            # Normal is perpendicular to edge
            normal = np.array([-edge[1], edge[0]])
            length = np.linalg.norm(normal)
            if length > 1e-6:
                axes.append(normal / length)
        return axes
    
    # Test all 4 separating axes (2 from each rectangle)
    all_axes = get_axes(corners1) + get_axes(corners2)
    
    for axis in all_axes:
        # Project all corners onto this axis
        proj1 = corners1 @ axis
        proj2 = corners2 @ axis
        
        # Check for gap between projections
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False  # Found a separating axis, no collision
    
    return True  # No separating axis found, collision!


def analyze_push_collisions(push_candidates: Dict) -> Dict:
    """
    Analyze collisions for both push options using OBB collision detection.
    
    This replaces the AABB-based collision detection with more accurate OBB.
    """
    opt_A = push_candidates.get('option_A', {})
    opt_B = push_candidates.get('option_B', {})
    other_zones = push_candidates.get('other_brick_zones', [])
    
    # Get fall parameters
    fall_A = opt_A.get('expected_fall_center', [0, 0])
    fall_B = opt_B.get('expected_fall_center', [0, 0])
    
    # Fall yaw is the L-axis direction of the fallen brick
    fall_yaw_deg = opt_A.get('expected_fall_yaw_deg', 0)
    fall_yaw = np.radians(fall_yaw_deg)
    
    # Brick dimensions
    half_L = BRICK_L / 2
    half_W = BRICK_W / 2
    
    a_collisions = []
    b_collisions = []
    
    print(f"    [OBB] Checking collisions with margin=2cm...")
    print(f"    [OBB] Fall yaw (L-axis): {fall_yaw_deg:.1f}°")
    
    for oz in other_zones:
        brick_pos = oz['position']
        brick_yaw = np.radians(oz['yaw_deg'])
        brick_idx = oz['index']
        
        # Check Option A collision using OBB
        collision_A = check_obb_collision(
            fall_A[0], fall_A[1], fall_yaw, half_L, half_W,
            brick_pos[0], brick_pos[1], brick_yaw, half_L, half_W,
            margin=0.02,  # 2cm safety margin
        )
        
        # Check Option B collision using OBB
        collision_B = check_obb_collision(
            fall_B[0], fall_B[1], fall_yaw, half_L, half_W,
            brick_pos[0], brick_pos[1], brick_yaw, half_L, half_W,
            margin=0.02,
        )
        
        if collision_A:
            a_collisions.append(brick_idx)
            print(f"    [OBB] Option A: COLLISION with #{brick_idx}")
        if collision_B:
            b_collisions.append(brick_idx)
            print(f"    [OBB] Option B: COLLISION with #{brick_idx}")
    
    a_safe = len(a_collisions) == 0
    b_safe = len(b_collisions) == 0
    
    if a_safe and not b_safe:
        recommended, reason = "A", f"A is safe, B collides with #{b_collisions}"
    elif b_safe and not a_safe:
        recommended, reason = "B", f"B is safe, A collides with #{a_collisions}"
    elif a_safe and b_safe:
        recommended, reason = "A", "Both options safe, defaulting to A"
    elif len(a_collisions) < len(b_collisions):
        recommended, reason = "A", f"A has fewer collisions ({len(a_collisions)} vs {len(b_collisions)})"
    elif len(b_collisions) < len(a_collisions):
        recommended, reason = "B", f"B has fewer collisions ({len(b_collisions)} vs {len(a_collisions)})"
    else:
        recommended, reason = "A", f"A has fewer collisions ({len(a_collisions)} vs {len(b_collisions)})"
    
    print(f"    [OBB] Result: A={'SAFE' if a_safe else 'COLLISION'}, B={'SAFE' if b_safe else 'COLLISION'}")
    print(f"    [OBB] Recommended: Option {recommended} ({reason})")
    
    return {
        "option_A": {"safe": a_safe, "collisions": a_collisions},
        "option_B": {"safe": b_safe, "collisions": b_collisions},
        "recommended": recommended,
        "recommendation_reason": reason,
    }


# ==================== Push Candidates Calculator ====================

def calculate_push_candidates(
    brick_position: List[float],
    brick_yaw: float,
    brick_pose: str,
    brick_size_LWH: Optional[List[float]] = None,
    ground_z: float = 0.86,
    other_bricks: Optional[List[Dict]] = None,
) -> Dict:
    """
    Calculate two candidate push directions for toppling a side/upright brick.
    
    brick_yaw is now in [0, π] range after refinement:
    - For UPRIGHT: yaw points along W axis (5cm edge of top surface W×H)
    - For SIDE: yaw points along L axis (11cm edge of top surface L×H)
    
    Push direction = perpendicular to yaw = along H axis direction
    
    After falling:
    - UPRIGHT: Was standing on W×H base, L vertical. Falls to L×W face.
      The L-axis of fallen brick is perpendicular to original W-axis.
      fallen_L_yaw = brick_yaw + 90° (perpendicular to W)
    - SIDE: Was lying on L×H base, W vertical. Falls to L×W face.
      The L-axis remains the same.
      fallen_L_yaw = brick_yaw (same as original L)
    """
    if brick_size_LWH is None:
        brick_size_LWH = [BRICK_L, BRICK_W, BRICK_H]
    
    L, W, H = brick_size_LWH
    half_L, half_W = L / 2, W / 2
    bx, by, bz = brick_position
    
    # Determine brick height based on pose
    if brick_pose == "upright":
        brick_height = L  # 11cm tall
    elif brick_pose == "side":
        brick_height = W  # 5cm tall
    else:
        brick_height = H
    
    # Calculate contact height
    raw_contact_height = brick_height * PUSH_CONTACT_HEIGHT_RATIO
    contact_height_above_ground = max(raw_contact_height, MIN_CONTACT_HEIGHT_ABOVE_GROUND)
    contact_height_absolute = ground_z + contact_height_above_ground
    
    # Push direction: perpendicular to refined yaw (along H axis)
    # brick_yaw points along W (upright) or L (side), push is perpendicular
    perp_x = -np.sin(brick_yaw)
    perp_y = np.cos(brick_yaw)
    
    # ===== Calculate fall offset and fallen brick L-axis direction =====
    if brick_pose == "upright":
        # UPRIGHT: L axis is vertical, standing on W×H base
        # Top surface is W×H, brick_yaw points along W axis
        # When it falls, it lands on L×W face
        # The L axis of fallen brick is PERPENDICULAR to original W axis
        fall_offset = L / 2 + 0.01  # 5.5cm + 1cm margin
        fallen_L_yaw = brick_yaw + np.pi / 2  # L axis perpendicular to W
        # Normalize to [0, π]
        while fallen_L_yaw >= np.pi:
            fallen_L_yaw -= np.pi
        while fallen_L_yaw < 0:
            fallen_L_yaw += np.pi
            
    elif brick_pose == "side":
        # SIDE: W axis is vertical, standing on L×H base
        # Top surface is L×H, brick_yaw points along L axis
        # When it falls, it lands on L×W face
        # The L axis remains the same direction
        fall_offset = W / 2 + 0.01  # 2.5cm + 1cm margin
        fallen_L_yaw = brick_yaw  # L axis unchanged
    else:
        fall_offset = 0
        fallen_L_yaw = brick_yaw
    
    print(f"    [PUSH] brick_yaw={np.degrees(brick_yaw):.1f}°, pose={brick_pose}")
    print(f"    [PUSH] Push direction: [{perp_x:.3f}, {perp_y:.3f}]")
    print(f"    [PUSH] Fallen L-axis yaw: {np.degrees(fallen_L_yaw):.1f}°")
    
    def compute_option(dir_sign: int) -> Dict:
        dx = dir_sign * perp_x
        dy = dir_sign * perp_y
        
        approach_x = bx - dx * PUSH_APPROACH_OFFSET
        approach_y = by - dy * PUSH_APPROACH_OFFSET
        approach_z = ground_z + SAFE_APPROACH_HEIGHT
        
        push_start_x = bx - dx * PUSH_START_OFFSET
        push_start_y = by - dy * PUSH_START_OFFSET
        
        push_end_x = push_start_x + dx * PUSH_DISTANCE_RECOMMENDED
        push_end_y = push_start_y + dy * PUSH_DISTANCE_RECOMMENDED
        
        fall_center_x = bx + dx * fall_offset
        fall_center_y = by + dy * fall_offset
        
        return {
            "push_direction": [float(dx), float(dy), 0.0],
            "approach_position": [float(approach_x), float(approach_y), float(approach_z)],
            "push_start_position": [float(push_start_x), float(push_start_y), float(contact_height_absolute)],
            "push_end_position": [float(push_end_x), float(push_end_y), float(contact_height_absolute)],
            "expected_fall_center": [float(fall_center_x), float(fall_center_y)],
            "expected_fall_yaw_deg": float(np.degrees(fallen_L_yaw)),
        }
    
    option_A = compute_option(+1)
    option_B = compute_option(-1)
    
    # Calculate occupied zones for other bricks (for OBB collision detection)
    other_brick_zones = []
    if other_bricks:
        for brick in other_bricks:
            idx = brick.get('index', 0)
            pos = brick.get('position', [0, 0, 0])
            yaw = brick.get('yaw', 0)
            
            # Normalize yaw to [0, π] for consistency
            while yaw < 0:
                yaw += np.pi
            while yaw >= np.pi:
                yaw -= np.pi
            
            other_brick_zones.append({
                'index': idx,
                'position': [pos[0], pos[1]],
                'yaw_deg': float(np.degrees(yaw)),
            })
    
    return {
        "target_brick_position": [float(bx), float(by), float(bz)],
        "target_brick_yaw": float(brick_yaw),
        "target_brick_yaw_deg": float(np.degrees(brick_yaw)),
        "target_brick_pose": brick_pose,
        "brick_height_cm": float(brick_height * 100),
        "contact_height": float(contact_height_absolute),
        "push_distance": float(PUSH_DISTANCE_RECOMMENDED),
        "option_A": option_A,
        "option_B": option_B,
        "other_brick_zones": other_brick_zones,
    }


# ==================== Head Camera Calculator ====================

class HeadCameraCalculator:
    """Head Camera Position Calculator"""
    
    def __init__(
        self, 
        intrinsics: Optional[Dict] = None,
        z_compensator: Optional[DynamicZCompensator] = None,
    ):
        if intrinsics is None:
            intrinsics = HEAD_INTRINSICS
        
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        self.z_compensator = z_compensator
        
        # Load static offset
        self.offset = np.zeros(3)
        offset_path = CALIB_DIR / "head_camera_offset.json"
        if offset_path.exists():
            try:
                with open(offset_path) as f:
                    data = json.load(f)
                    self.offset = np.array(data.get('offset_xyz', [0, 0, 0]))
                    print(f"[HeadCalc] Static offset: X={self.offset[0]:+.4f}, Y={self.offset[1]:+.4f}, Z={self.offset[2]:+.4f}")
            except Exception as e:
                print(f"[HeadCalc] Failed to load offset: {e}")
        
        if self.z_compensator is not None and self.z_compensator.enabled:
            print(f"[HeadCalc] DynamicZCompensator: ENABLED ({self.z_compensator.model_type})")
        else:
            print("[HeadCalc] DynamicZCompensator: DISABLED")
    
    def compute(
        self, 
        mask: np.ndarray, 
        depth: np.ndarray, 
        tf_matrix: np.ndarray
    ) -> Optional[Dict]:
        """Compute brick information in base_link frame."""
        # Import yaw estimation from sam3_segmenter
        from perception.sam3_segmenter import estimate_orientation_from_mask
        
        h, w = depth.shape
        
        # Handle mask dimensions
        mask = mask[0] if len(mask.shape) == 3 else mask
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        mask_bool = mask > 0.5
        if not np.any(mask_bool):
            return None
        
        # Get mask centroid
        ys, xs = np.where(mask_bool)
        px, py = np.mean(xs), np.mean(ys)
        
        # Estimate orientation using sam3_segmenter
        yaw_cam = estimate_orientation_from_mask(mask_bool)
        
        # Get depth value
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=2)
        valid = depth[eroded > 0] if np.any(eroded) else depth[mask_bool]
        valid = valid[valid > 0]
        
        if len(valid) == 0:
            return None
        
        z = np.median(valid) / 1000.0
        depth_median_m = z
        z = np.clip(z, 0.1, 2.0)
        
        # Calculate real area
        area_cm2, area_pixels = calculate_real_area(mask_bool, depth, self.fx, self.fy)
        
        # Compute 3D position in camera frame
        pos_cam = np.array([
            (px - self.cx) * z / self.fx,
            (py - self.cy) * z / self.fy,
            z
        ])
        
        # Transform to base_link frame
        pos_base = (tf_matrix @ np.append(pos_cam, 1))[:3] + self.offset
        
        # Transform yaw to base frame
        yaw_base = yaw_cam + np.arctan2(tf_matrix[1, 0], tf_matrix[0, 0]) + np.pi
        
        # Apply dynamic Z compensation
        z_compensation = 0.0
        if self.z_compensator is not None and self.z_compensator.enabled:
            z_compensation = self.z_compensator.compute_compensation(pos_base[0], pos_base[1])
            pos_base[2] += z_compensation
            print(f"[HeadCalc] Z compensation: {z_compensation*1000:+.1f} mm at X={pos_base[0]:.3f}, Y={pos_base[1]:.3f}")
        
        # Normalize yaw
        while yaw_base > np.pi:
            yaw_base -= 2 * np.pi
        while yaw_base < -np.pi:
            yaw_base += 2 * np.pi
        
        return {
            'position': pos_base, 
            'yaw': yaw_base,
            'area_cm2': area_cm2,
            'area_pixels': area_pixels,
            'z_compensation': z_compensation,
            'depth_median_m': depth_median_m,
        }


# ==================== Hand-Eye Camera Calculator ====================

class HandEyeCalculator:
    """Hand-Eye Camera Position Calculator"""
    
    def __init__(self, intrinsics: Dict, extrinsics: Dict):
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        
        self.T_cam2gripper = np.eye(4)
        self.T_cam2gripper[:3, :3] = np.array(extrinsics['rotation_matrix'])
        self.T_cam2gripper[:3, 3] = np.array(extrinsics['translation'])
    
    def compute(
        self, 
        mask: np.ndarray, 
        shape: Tuple[int, int],
        R_g2b: np.ndarray, 
        t_g2b: np.ndarray,
        reference_z: float, 
        reference_yaw: float,
        depth_mm: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """Compute refined brick position.
        
        If depth_mm is provided, uses real depth for XYZ (more accurate).
        Otherwise falls back to ray-plane intersection with reference_z.
        """
        from perception.sam3_segmenter import estimate_orientation_from_mask
        
        h, w = shape
        
        mask = mask[0] if len(mask.shape) == 3 else mask
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        mask_bool = mask > 0.5
        if not np.any(mask_bool):
            return None
        
        ys, xs = np.where(mask_bool)
        px, py = np.mean(xs), np.mean(ys)
        
        yaw_cam = estimate_orientation_from_mask(mask_bool)
        
        T_g2b = np.eye(4)
        T_g2b[:3, :3] = R_g2b
        T_g2b[:3, 3] = t_g2b.flatten()
        
        T_cam2base = T_g2b @ self.T_cam2gripper
        
        R_mat = T_cam2base[:3, :3]
        t_vec = T_cam2base[:3, 3]
        
        # ---- Z estimation: prefer real depth over ray-plane ----
        if depth_mm is not None:
            # Sample depth at mask center (median of mask region for robustness)
            mask_depths = depth_mm[mask_bool]
            valid = mask_depths[mask_depths > 0]
            if len(valid) > 0:
                z_cam = float(np.median(valid)) / 1000.0  # mm → m
            else:
                # Fallback: ray-plane intersection
                nx = (px - self.cx) / self.fx
                ny = (py - self.cy) / self.fy
                coeff = R_mat[2, 0] * nx + R_mat[2, 1] * ny + R_mat[2, 2]
                z_cam = (reference_z - t_vec[2]) / coeff if abs(coeff) > 1e-6 else 0.3
        else:
            # Original: ray-plane intersection using reference_z
            nx = (px - self.cx) / self.fx
            ny = (py - self.cy) / self.fy
            coeff = R_mat[2, 0] * nx + R_mat[2, 1] * ny + R_mat[2, 2]
            z_cam = (reference_z - t_vec[2]) / coeff if abs(coeff) > 1e-6 else 0.3
        
        z_cam = max(0.05, min(1.0, z_cam))
        
        pos_cam = np.array([
            (px - self.cx) * z_cam / self.fx,
            (py - self.cy) * z_cam / self.fy,
            z_cam
        ])
        
        pos_base = (T_cam2base @ np.append(pos_cam, 1))[:3]
        yaw_base = yaw_cam + np.arctan2(T_cam2base[1, 0], T_cam2base[0, 0])
        
        while yaw_base > np.pi:
            yaw_base -= 2 * np.pi
        while yaw_base < -np.pi:
            yaw_base += 2 * np.pi
        
        diff = yaw_base - reference_yaw
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        if abs(diff) > np.pi / 2:
            yaw_base = yaw_base - np.pi if yaw_base > 0 else yaw_base + np.pi
        
        return {'position': pos_base, 'yaw': yaw_base}

# ==================== Factory Functions ====================

def create_head_calculator(
    intrinsics: Optional[Dict] = None,
    calib_dir: Optional[Path] = None,
    enable_z_compensation: bool = True,
) -> HeadCameraCalculator:
    if calib_dir is None:
        calib_dir = CALIB_DIR
    
    z_compensator = None
    if enable_z_compensation:
        offset_path = calib_dir / "head_camera_offset.json"
        z_compensator = DynamicZCompensator(offset_path)
    
    return HeadCameraCalculator(intrinsics=intrinsics, z_compensator=z_compensator)


def create_handeye_calculator(
    side: str = "left",
    calib_dir: Optional[Path] = None
) -> HandEyeCalculator:
    if calib_dir is None:
        calib_dir = CALIB_DIR
    
    intr_path = calib_dir / f"hand_eye_intrinsics_{side}.json"
    extr_path = calib_dir / f"hand_eye_extrinsics_{side}.json"
    
    with open(intr_path) as f:
        intrinsics = json.load(f)
    with open(extr_path) as f:
        extrinsics = json.load(f)
    
    return HandEyeCalculator(intrinsics, extrinsics)