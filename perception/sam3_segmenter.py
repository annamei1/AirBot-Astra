"""
SAM3 Segmentation Module

Provides:
- SAM3 text-prompted segmentation (singleton, thread-safe)
- Top surface extraction for upright/side bricks
- Rectangle fitting for yaw estimation

Singleton pattern for efficient model loading and thread-safe segmentation.
"""

# ===== Setup SAM3 path FIRST (before any SAM3 imports) =====
import os
import sys
# Where the SAM3 repository is checked out: $SAM3_HOME, or ~/sam3-main by default.
SAM3_HOME = os.path.expanduser(os.environ.get("SAM3_HOME", "~/sam3-main"))
sys.path.insert(0, SAM3_HOME)

# ===== Suppress warnings =====
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import logging
logging.getLogger("root").setLevel(logging.WARNING)

# ===== Standard imports =====
import cv2
import gc
import torch
import threading
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple, List

# ===== SAM3 imports (after path setup) =====
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ==================== Top Surface Extraction ====================

def extract_top_surface_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_threshold_mm: float = 15.0,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Extract top surface region from mask using depth information.
    
    For upright/side bricks, the SAM3 mask includes both top face and side faces.
    This function uses depth to isolate only the top surface (closest to camera).
    
    Args:
        mask: SAM3 segmentation mask (H, W) or (1, H, W)
        depth: Depth image (H, W) in millimeters
        depth_threshold_mm: Points within this depth of minimum are considered "top"
        
    Returns:
        top_mask: Binary mask of top surface only (H, W), or None if failed
        info: Dict with extraction statistics
    """
    h, w = depth.shape
    
    # Handle mask dimensions
    if len(mask.shape) == 3:
        mask = mask[0]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), 
                         interpolation=cv2.INTER_NEAREST)
    
    mask_bool = mask > 0.5
    if not np.any(mask_bool):
        return None, {'error': 'Empty mask'}
    
    # Get all points within mask
    ys, xs = np.where(mask_bool)
    depths_at_mask = depth[ys, xs].astype(np.float32)
    
    # Filter valid depth values (10cm - 2m range)
    valid_mask = (depths_at_mask > 100) & (depths_at_mask < 2000)
    if np.sum(valid_mask) < 20:
        return None, {'error': 'Not enough valid depth points'}
    
    valid_depths = depths_at_mask[valid_mask]
    valid_ys = ys[valid_mask]
    valid_xs = xs[valid_mask]
    
    # Find minimum depth (top surface - closest to camera)
    min_depth = np.min(valid_depths)
    
    # Select points within threshold of minimum depth
    top_mask_indices = (valid_depths - min_depth) <= depth_threshold_mm
    top_ys = valid_ys[top_mask_indices]
    top_xs = valid_xs[top_mask_indices]
    
    if len(top_xs) < 10:
        return None, {'error': 'Too few points in top surface',
                     'min_depth': float(min_depth),
                     'threshold': depth_threshold_mm}
    
    # Create top surface mask
    top_mask = np.zeros((h, w), dtype=np.uint8)
    top_mask[top_ys, top_xs] = 255
    
    # Morphological closing to fill small gaps
    kernel = np.ones((3, 3), np.uint8)
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, kernel)
    
    # Compute statistics
    top_depth_mean = np.mean(valid_depths[top_mask_indices])
    top_depth_std = np.std(valid_depths[top_mask_indices])
    top_area_pixels = np.sum(top_mask > 0)
    
    info = {
        'min_depth_mm': float(min_depth),
        'top_depth_mean_mm': float(top_depth_mean),
        'top_depth_std_mm': float(top_depth_std),
        'depth_threshold_mm': depth_threshold_mm,
        'top_area_pixels': int(top_area_pixels),
        'total_mask_pixels': int(np.sum(mask_bool)),
        'top_ratio': float(top_area_pixels / np.sum(mask_bool)) if np.sum(mask_bool) > 0 else 0,
    }
    
    return top_mask, info


def fit_rectangle_to_mask(
    mask: np.ndarray,
) -> Tuple[Optional[Tuple], Dict]:
    """
    Fit minimum area rectangle to a binary mask.
    
    Returns the rectangle info and the long edge direction.
    
    Args:
        mask: Binary mask (H, W), values 0 or 255
        
    Returns:
        rect: ((cx, cy), (w, h), angle) from cv2.minAreaRect, or None
        info: Dict with rectangle information including long_axis_angle_deg
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None, {'error': 'No contours found'}
    
    # Use largest contour
    largest = max(contours, key=cv2.contourArea)
    
    if len(largest) < 5:
        return None, {'error': 'Contour too small'}
    
    # Fit minimum area rectangle
    rect = cv2.minAreaRect(largest)
    (cx, cy), (rect_w, rect_h), angle = rect
    
    # Determine long and short edges
    if rect_w >= rect_h:
        long_edge = rect_w
        short_edge = rect_h
        long_axis_angle = angle
    else:
        long_edge = rect_h
        short_edge = rect_w
        long_axis_angle = angle + 90
    
    # Compute aspect ratio
    aspect_ratio = long_edge / short_edge if short_edge > 0 else 0
    
    info = {
        'rect_center': (float(cx), float(cy)),
        'rect_size': (float(rect_w), float(rect_h)),
        'rect_angle': float(angle),
        'long_edge_px': float(long_edge),
        'short_edge_px': float(short_edge),
        'aspect_ratio': float(aspect_ratio),
        'long_axis_angle_deg': float(long_axis_angle),
    }
    
    return rect, info


def extract_top_surface_yaw(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_threshold_mm: float = 15.0,
    min_aspect_ratio: float = 1.3,
) -> Tuple[Optional[float], Dict]:
    """
    Extract yaw angle from the top surface of a mask.
    
    This is the main function for refining yaw of upright/side bricks.
    It extracts the top surface using depth, fits a rectangle, and returns
    the long edge direction as yaw.
    
    Args:
        mask: SAM3 segmentation mask (H, W) or (1, H, W)
        depth: Depth image (H, W) in millimeters
        depth_threshold_mm: Depth threshold for top surface extraction
        min_aspect_ratio: Minimum long/short ratio to trust the result
        
    Returns:
        yaw: Long edge angle in radians (normalized to [-pi/2, pi/2]), or None if failed
        info: Dict with extraction and fitting details
    """
    # Step 1: Extract top surface
    top_mask, extract_info = extract_top_surface_mask(mask, depth, depth_threshold_mm)
    
    if top_mask is None:
        return None, {'error': f"Top surface extraction failed: {extract_info.get('error', 'unknown')}",
                     'extract_info': extract_info}
    
    # Step 2: Fit rectangle
    rect, rect_info = fit_rectangle_to_mask(top_mask)
    
    if rect is None:
        return None, {'error': f"Rectangle fitting failed: {rect_info.get('error', 'unknown')}",
                     'extract_info': extract_info, 'rect_info': rect_info}
    
    # Step 3: Check aspect ratio confidence
    aspect_ratio = rect_info['aspect_ratio']
    
    if aspect_ratio < min_aspect_ratio:
        return None, {'error': f"Aspect ratio {aspect_ratio:.2f} < {min_aspect_ratio}",
                     'extract_info': extract_info, 'rect_info': rect_info}
    
    # Step 4: Convert angle to radians
    long_axis_angle_deg = rect_info['long_axis_angle_deg']
    yaw = -np.radians(long_axis_angle_deg)
    
    # Normalize to [-pi/2, pi/2]
    while yaw > np.pi / 2:
        yaw -= np.pi
    while yaw < -np.pi / 2:
        yaw += np.pi
    
    yaw = float(np.clip(yaw, -np.pi / 2, np.pi / 2))
    
    info = {
        'yaw_rad': yaw,
        'yaw_deg': float(np.degrees(yaw)),
        'long_axis_angle_deg': long_axis_angle_deg,
        'aspect_ratio': aspect_ratio,
        'extract_info': extract_info,
        'rect_info': rect_info,
        'top_mask': top_mask,  # Include for visualization
    }
    
    return yaw, info


def estimate_orientation_from_mask(mask: np.ndarray) -> float:
    """
    Estimate orientation angle (yaw) from binary mask using minAreaRect.
    
    This is the standard method for flat bricks where we use the full mask.
    
    Args:
        mask: binary mask (H, W)
        
    Returns:
        yaw angle in radians, normalized to [-pi/2, pi/2]
    """
    # Handle mask dimensions
    if len(mask.shape) == 3:
        mask = mask[0]
    
    mask_bool = mask > 0.5
    if not np.any(mask_bool):
        return 0.0
    
    contours, _ = cv2.findContours(
        (mask_bool * 255).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return 0.0
    
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0
    
    rect = cv2.minAreaRect(largest)
    (_, _), (w, h), angle = rect
    
    # Determine long axis angle
    long_angle = angle + 90 if w < h else angle
    yaw = -np.radians(long_angle)
    
    # Normalize to [-pi/2, pi/2]
    while yaw > np.pi / 2:
        yaw -= np.pi
    while yaw < -np.pi / 2:
        yaw += np.pi
    
    return float(np.clip(yaw, -np.pi / 2, np.pi / 2))


# ==================== SAM3 Segmenter Class ====================

class SAM3Segmenter:
    """
    SAM3 Segmenter (Singleton, Thread-safe)
    
    Provides:
    - Text-prompted segmentation
    - Top surface extraction for upright/side bricks
    - Yaw estimation from masks
    
    Usage:
        segmenter = SAM3Segmenter("/path/to/checkpoint.pt")
        masks = segmenter.segment(image, "brick")
        yaw, info = segmenter.get_top_surface_yaw(mask, depth)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, checkpoint_path: str, confidence: float = 0.5):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, checkpoint_path: str, confidence: float = 0.5):
        if self._initialized:
            return
        
        print("[SAM3] Loading model...")
        
        self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        self.processor = Sam3Processor(
            self.model, 
            resolution=1008, 
            confidence_threshold=confidence
        )
        self._segment_lock = threading.Lock()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        self._initialized = True
        print("[SAM3] Model loaded")
    
    def segment(self, img_bgr: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """
        Segment image with text prompt.
        
        Args:
            img_bgr: BGR image (OpenCV format)
            prompt: text prompt for segmentation
            
        Returns:
            Segmentation masks (N, H, W) or None if no detection
        """
        with self._segment_lock:
            # Convert BGR to RGB PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
            # Process
            state = self.processor.set_image(pil_img)
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
            
            # Extract masks
            masks = None
            if out["masks"] is not None:
                masks = out["masks"].cpu().numpy()
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            return masks
    
    def segment_box(
        self,
        img_bgr: np.ndarray,
        box_xyxy: Tuple[float, float, float, float],
        prompt: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Class-agnostic segmentation from a pixel box (additive; nothing else uses it).

        SAM3 accepts a geometric prompt without any text — it substitutes an internal "visual"
        prompt and relies on the box alone. That is what makes this usable on objects whose name
        the text encoder does not resolve.

        Args:
            img_bgr: BGR image
            box_xyxy: (x0, y0, x1, y1) in pixels
            prompt: optional text to combine with the box

        Returns:
            Segmentation masks (N, H, W) or None
        """
        with self._segment_lock:
            h, w = img_bgr.shape[:2]
            x0, y0, x1, y1 = (float(v) for v in box_xyxy)
            x0, x1 = sorted((max(0.0, x0), min(float(w), x1)))
            y0, y1 = sorted((max(0.0, y0), min(float(h), y1)))
            if x1 - x0 < 2 or y1 - y0 < 2:
                return None
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            state = self.processor.set_image(pil_img)
            if prompt:
                self.processor.set_text_prompt(state=state, prompt=prompt)
            # SAM3 wants [center_x, center_y, width, height], normalised to [0, 1]
            out = self.processor.add_geometric_prompt(
                [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h], True, state)
            masks = out["masks"].cpu().numpy() if out.get("masks") is not None else None
            torch.cuda.empty_cache()
            return masks

    def segment_rgb(self, img_rgb: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """
        Segment RGB image with text prompt.
        
        Args:
            img_rgb: RGB image
            prompt: text prompt
            
        Returns:
            Segmentation masks or None
        """
        with self._segment_lock:
            pil_img = Image.fromarray(img_rgb)
            state = self.processor.set_image(pil_img)
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
            
            masks = None
            if out["masks"] is not None:
                masks = out["masks"].cpu().numpy()
            
            torch.cuda.empty_cache()
            return masks
    
    # ==================== Yaw Estimation Methods ====================
    
    def get_mask_yaw(self, mask: np.ndarray) -> float:
        """
        Get yaw from mask using standard minAreaRect method.
        
        Suitable for flat bricks where the full mask represents the top face.
        
        Args:
            mask: Segmentation mask (H, W) or (1, H, W)
            
        Returns:
            Yaw angle in radians
        """
        return estimate_orientation_from_mask(mask)
    
    def get_top_surface_yaw(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        depth_threshold_mm: float = 15.0,
        min_aspect_ratio: float = 1.3,
    ) -> Tuple[Optional[float], Dict]:
        """
        Get yaw from the top surface of a mask using depth.
        
        Suitable for upright/side bricks where the mask includes side faces.
        This extracts only the top surface and fits a rectangle to get
        the long edge direction.
        
        Args:
            mask: Segmentation mask (H, W) or (1, H, W)
            depth: Depth image (H, W) in millimeters
            depth_threshold_mm: Depth threshold for top surface
            min_aspect_ratio: Minimum aspect ratio to trust result
            
        Returns:
            yaw: Angle in radians, or None if extraction failed
            info: Dict with extraction details
        """
        return extract_top_surface_yaw(mask, depth, depth_threshold_mm, min_aspect_ratio)
    
    def extract_top_surface(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        depth_threshold_mm: float = 15.0,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Extract the top surface mask from a full brick mask.
        
        Args:
            mask: Full brick mask
            depth: Depth image in mm
            depth_threshold_mm: Depth threshold
            
        Returns:
            top_mask: Top surface mask (H, W) with values 0 or 255
            info: Extraction statistics
        """
        return extract_top_surface_mask(mask, depth, depth_threshold_mm)
    
    def fit_rectangle(self, mask: np.ndarray) -> Tuple[Optional[Tuple], Dict]:
        """
        Fit minimum area rectangle to a mask.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            rect: ((cx, cy), (w, h), angle) or None
            info: Rectangle information
        """
        return fit_rectangle_to_mask(mask)
    
    # ==================== Visualization Methods ====================

    def draw_detection(
        self, 
        frame: np.ndarray, 
        mask: Optional[np.ndarray],
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        selected_index: Optional[int] = None,
        draw_contour: bool = True,
    ) -> np.ndarray:
        """
        Draw detection on frame.
        
        Args:
            frame: BGR image to draw on
            mask: segmentation mask(s) from segment() - can be single or multiple
            color: box color (BGR) for selected/single detection
            thickness: line thickness
            selected_index: if provided, highlight this mask specially (others drawn dimmer)
            draw_contour: if True, draw actual contour; if False, draw minAreaRect
            
        Returns:
            Frame with detection visualization
        """
        if mask is None:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Color palette for multiple detections
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 165, 255),  # Orange
            (255, 255, 0),  # Cyan
        ]
        
        # Handle single vs multiple masks
        if len(mask.shape) == 2:
            masks_list = [mask]
        elif len(mask.shape) == 3:
            if mask.shape[0] == 1:
                masks_list = [mask[0]]
            else:
                masks_list = [mask[i] for i in range(mask.shape[0])]
        elif len(mask.shape) == 4:
            masks_list = [mask[i, 0] for i in range(mask.shape[0])]
        else:
            return result
        
        # Draw each mask
        for i, m in enumerate(masks_list):
            # Resize if needed
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            
            mask_bool = (m > 0.5).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            largest = max(contours, key=cv2.contourArea)
            
            # Determine color and thickness for this mask
            if selected_index is not None:
                if i == selected_index:
                    draw_color = (0, 255, 0)  # Bright green
                    draw_thickness = 3
                else:
                    draw_color = (100, 100, 100)  # Gray
                    draw_thickness = 1
            else:
                draw_color = colors[i % len(colors)]
                draw_thickness = thickness
            
            # Draw contour or bounding box
            if draw_contour:
                cv2.drawContours(result, [largest], -1, draw_color, draw_thickness)
            else:
                if len(largest) >= 5:
                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(result, [box], 0, draw_color, draw_thickness)
            
            # Draw center point and index
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                if selected_index is None or i == selected_index:
                    cv2.circle(result, (cx, cy), 5, draw_color, -1)
                    cv2.circle(result, (cx, cy), 7, (255, 255, 255), 2)
                    cv2.putText(result, f"#{i}", (cx + 10, cy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
                else:
                    cv2.circle(result, (cx, cy), 3, draw_color, -1)
                    cv2.putText(result, f"#{i}", (cx + 10, cy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)
        
        return result
    
    def draw_top_surface(
        self,
        frame: np.ndarray,
        top_mask: np.ndarray,
        rect_info: Dict,
        color: tuple = (0, 255, 255),
        draw_arrow: bool = True,
    ) -> np.ndarray:
        """
        Draw top surface visualization with rectangle and yaw direction.
        
        Args:
            frame: BGR image to draw on
            top_mask: Top surface mask (H, W)
            rect_info: Rectangle info from fit_rectangle_to_mask
            color: Drawing color
            draw_arrow: Whether to draw yaw direction arrow
            
        Returns:
            Frame with visualization
        """
        result = frame.copy()
        
        # Draw top surface region (semi-transparent)
        mask_overlay = result.copy()
        mask_color = tuple(c // 3 for c in color)
        mask_overlay[top_mask > 0] = mask_color
        result = cv2.addWeighted(result, 0.7, mask_overlay, 0.3, 0)
        
        # Draw fitted rectangle
        cx, cy = rect_info['rect_center']
        rect_size = rect_info['rect_size']
        rect_angle = rect_info['rect_angle']
        
        rect = ((cx, cy), rect_size, rect_angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(result, [box], 0, color, 2)
        
        # Draw long axis direction (yaw)
        if draw_arrow:
            long_axis_angle_deg = rect_info['long_axis_angle_deg']
            angle_rad = np.radians(-long_axis_angle_deg)
            arrow_len = 40
            end_x = int(cx + arrow_len * np.cos(angle_rad))
            end_y = int(cy + arrow_len * np.sin(angle_rad))
            cv2.arrowedLine(result, (int(cx), int(cy)), (end_x, end_y), 
                           (0, 0, 255), 2, tipLength=0.3)
        
        # Add label
        label = f"yaw:{rect_info['long_axis_angle_deg']:.1f}° ratio:{rect_info['aspect_ratio']:.2f}"
        cv2.putText(result, label, (int(cx) + 10, int(cy) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result


def create_segmenter(
    checkpoint_path: str = os.path.join(SAM3_HOME, "checkpoint", "sam3.pt"),
    confidence: float = 0.5
) -> SAM3Segmenter:
    """
    Create or get SAM3 segmenter instance.
    
    Args:
        checkpoint_path: path to SAM3 checkpoint
        confidence: confidence threshold
        
    Returns:
        SAM3Segmenter instance
    """
    return SAM3Segmenter(checkpoint_path, confidence)