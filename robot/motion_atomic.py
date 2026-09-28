"""
Atomic Motion Actions for Robot Control

Provides low-level atomic actions that can be composed into task sequences.
Each action is independent and can be called in any order.

Atomic Actions:
1. detect_head       - Head camera detection
2. detect_handeye    - Hand-eye camera fine positioning
3. move_to_position  - Move arm to XYZ + Yaw
4. descend_to_z      - Descend to specified Z height
5. lift_by           - Lift by specified height
6. open_gripper      - Open gripper to gap
7. close_gripper     - Close gripper with optional grasp check
8. release_with_contact - Closed-loop release with contact detection
"""

import time
import numpy as np
from typing import Optional, Dict, Tuple, List, Callable, Any


def _tilt(pitch: float, roll: float) -> dict:
    """Extra move_arm kwargs for a tilted tool, and nothing at all when the tool points straight down.

    Passing pitch/roll only when they are non-zero keeps this executor usable with any robot_env that
    predates them, including the MMK2 RealRobotEnv this file was originally written against.
    """
    return {} if (pitch == 0.0 and roll == 0.0) else {"pitch": float(pitch), "roll": float(roll)}


class AtomicMotionExecutor:
    """
    Low-level atomic motion executor.
    
    Provides independent atomic actions that can be composed into
    higher-level task sequences by MotionExecutor.
    """
    
    def __init__(
        self,
        robot_env,
        segmenter,
        head_calc,
        handeye_calc,
        llm_planner=None,
        prompt: str = "block, brick, rectangular object",
    ):
        """
        Initialize atomic executor with required components.
        
        Args:
            robot_env: RobotEnv instance for robot control
            segmenter: SAM3 segmenter for brick detection
            head_calc: Head camera position calculator
            handeye_calc: Hand-eye camera position calculator
            llm_planner: LLM grasp planner
            prompt: Segmentation prompt for brick detection
        """
        self.robot_env = robot_env
        self.segmenter = segmenter
        self.head_calc = head_calc
        self.handeye_calc = handeye_calc
        self.llm_planner = llm_planner
        self.prompt = prompt
        # Max press depth used by release_with_contact (current-protected). Task-configurable:
        # 15 mm for placing into a bowl / on the table, 5 mm when stacking on other bricks.
        self.release_max_depth: float = 0.015
        
        # Callbacks (set by parent MotionExecutor)
        self._step_callback: Optional[Callable[[str], None]] = None
        self._result_callback: Optional[Callable[[str, Dict], None]] = None
        self._mask_callback: Optional[Callable[[str, np.ndarray], None]] = None
        self._check_abort: Callable[[], bool] = lambda: False
        
        # Execution context - shared state between atomic actions
        self._context: Dict[str, Any] = {}
    
    # ==================== Callback Management ====================
    
    def set_callbacks(
        self,
        step_callback: Optional[Callable[[str], None]] = None,
        result_callback: Optional[Callable[[str, Dict], None]] = None,
        mask_callback: Optional[Callable[[str, np.ndarray], None]] = None,
        check_abort: Optional[Callable[[], bool]] = None,
    ):
        """Set callbacks for UI integration."""
        if step_callback:
            self._step_callback = step_callback
        if result_callback:
            self._result_callback = result_callback
        if mask_callback:
            self._mask_callback = mask_callback
        if check_abort:
            self._check_abort = check_abort
    
    def _log_step(self, message: str):
        """Log step message."""
        print(message)
        if self._step_callback:
            self._step_callback(message)
    
    def _save_result(self, key: str, result: Dict):
        """Save result via callback."""
        if self._result_callback:
            self._result_callback(key, result)
    
    def _save_mask(self, key: str, mask: np.ndarray):
        """Save mask via callback."""
        if self._mask_callback:
            self._mask_callback(key, mask)
    
    # ==================== Context Management ====================
    
    def clear_context(self):
        """Clear execution context."""
        self._context = {}
    
    def get_context(self) -> Dict[str, Any]:
        """Get current execution context."""
        return self._context.copy()
    
    def set_context(self, key: str, value: Any):
        """Set a context value."""
        self._context[key] = value
    
    # ==================== Atomic Action 1: Head Camera Detection ====================
    
    def detect_head(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        select_index: int = 0,
        log_step: bool = True,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        [Atomic] Detect brick(s) using head camera.
        """
        if log_step:
            self._log_step("\n[Atomic: detect_head] Head camera detection...")
        
        if self._check_abort():
            return False, None, "Aborted"
        
        tf_matrix = self.robot_env.get_head_camera_transform()
        if tf_matrix is None:
            return False, None, "TF failed"
        
        masks = self.segmenter.segment(rgb, self.prompt)
        if self._check_abort():
            return False, None, "Aborted"
        if masks is None or len(masks) == 0:
            return False, None, "No brick detected"
        
        if select_index >= len(masks):
            select_index = 0
        
        head_result = self.head_calc.compute(masks[select_index], depth, tf_matrix)
        if head_result is None:
            return False, None, "Position calculation failed"
        
        self._save_mask('head', masks[select_index])
        
        brick_pos = head_result['position']
        brick_yaw = head_result['yaw']
        z_compensation = head_result.get('z_compensation', 0.0)
        
        result = {
            'position': brick_pos,
            'yaw': brick_yaw,
            'z_compensation': z_compensation,
            'selected_index': select_index,
            'total_detected': len(masks),
        }
        
        # 始终计算所有砖块的位置，不再限制 len(masks) > 1
        all_bricks = []
        for i, mask in enumerate(masks):
            r = self.head_calc.compute(mask, depth, tf_matrix)
            if r is not None:
                all_bricks.append({
                    'position': r['position'].tolist() if hasattr(r['position'], 'tolist') else list(r['position']),
                    'yaw': float(r['yaw']),
                    'z_compensation': r.get('z_compensation', 0.0),
                })
        result['all_bricks'] = all_bricks
        
        # Update context
        self._context['head_result'] = result
        self._context['brick_position'] = brick_pos.tolist() if hasattr(brick_pos, 'tolist') else list(brick_pos)
        self._context['brick_yaw'] = float(brick_yaw)
        self._context['z_compensation'] = z_compensation
        
        self._save_result('head', result)
        
        if log_step:
            print(f"  Brick: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}] m, yaw={np.degrees(brick_yaw):.1f}°")
            if abs(z_compensation) > 0.001:
                print(f"  Z compensation: {z_compensation*1000:+.1f} mm")
            print(f"  Total detected: {len(masks)}")
        
        return True, result, None
    
    # ==================== Atomic Action 2: Move to Position ====================
    
    def move_to_position(
        self,
        target_xyz: List[float],
        target_yaw: float,
        wait: float = 2.5,
        log_step: bool = True,
        step_name: str = "move_to_position",
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        [Atomic] Move arm to specified position.
        
        Args:
            target_xyz: [x, y, z] target position
            target_yaw: target yaw angle (radians)
            wait: wait time after move (seconds)
            log_step: Whether to log step message
            step_name: Name to show in log
            
        Returns:
            (success, result_dict, error_message)
        """
        if log_step:
            self._log_step(f"\n[Atomic: {step_name}] Moving to position...")
            print(f"  Target: [{target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f}] m, yaw={np.degrees(target_yaw):.1f}°")
        
        if self._check_abort():
            return False, None, "Aborted"
        
        if not self.robot_env.move_arm(target_xyz, target_yaw, wait=wait, check_abort=self._check_abort,
                                       **_tilt(pitch, roll)):
            return False, None, "Move to position failed"
        
        self._context['last_move_position'] = list(target_xyz)
        self._context['last_move_yaw'] = target_yaw
        
        if log_step:
            print(f"  Reached position")
        
        return True, {'position': list(target_xyz), 'yaw': target_yaw}, None
    
    # ==================== Atomic Action 3: Hand-eye Detection ====================
    
    def detect_handeye(
        self,
        reference_z: float,
        reference_yaw: float,
        reference_xy: Optional[List[float]] = None,
        offset_xy: Optional[List[float]] = None,
        z_compensation: float = 0.0,
        log_step: bool = True,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        [Atomic] Fine positioning using hand-eye camera (XYZ + Yaw).
        
        Uses wrist D405 depth for accurate Z estimation.
        """
        if log_step:
            self._log_step("\n[Atomic: detect_handeye] Hand-eye fine positioning...")
        
        if self._check_abort():
            return False, None, "Aborted"
        
        # ★ 获取 RGB + Depth（改为 tuple 返回）
        handeye_frame = self.robot_env.get_handeye_camera_frame()
        if isinstance(handeye_frame, tuple):
            handeye_img, handeye_depth = handeye_frame
        else:
            # 兼容旧版只返回 RGB 的情况
            handeye_img = handeye_frame
            handeye_depth = None
        
        arm_pose = self.robot_env.get_arm_pose()
        if handeye_img is None or arm_pose is None:
            return False, None, "Cannot get hand-eye data"
        
        masks = self.segmenter.segment(handeye_img, self.prompt)
        if self._check_abort():
            return False, None, "Aborted"
        if masks is None or len(masks) == 0:
            return False, None, "Hand-eye detection failed"
        
        # Select closest mask to reference position
        selected_mask_idx = 0
        if len(masks) > 1 and reference_xy is not None:
            if log_step:
                print(f"  Detected {len(masks)} objects, selecting closest...")
            
            min_dist = float('inf')
            for i, mask in enumerate(masks):
                result = self.handeye_calc.compute(
                    mask, handeye_img.shape[:2], arm_pose[0], arm_pose[1],
                    reference_z=reference_z, reference_yaw=reference_yaw,
                    depth_mm=handeye_depth,
                )
                if result is not None:
                    pos = result['position']
                    dist = np.sqrt((pos[0] - reference_xy[0])**2 + (pos[1] - reference_xy[1])**2)
                    if log_step:
                        print(f"    Mask {i}: dist={dist:.4f} m")
                    if dist < min_dist:
                        min_dist = dist
                        selected_mask_idx = i
            
            if log_step:
                print(f"  Selected mask #{selected_mask_idx}")
        
        handeye_result = self.handeye_calc.compute(
            masks[selected_mask_idx], handeye_img.shape[:2], arm_pose[0], arm_pose[1],
            reference_z=reference_z, reference_yaw=reference_yaw,
            depth_mm=handeye_depth,
        )
        
        self._save_mask('handeye', masks[selected_mask_idx])
        self._context['last_handeye_image'] = handeye_img.copy()
        self._context['last_handeye_mask'] = masks[selected_mask_idx].copy() 
               
        if handeye_result is None:
            return False, None, "Hand-eye calculation failed"
        
        self._save_result('handeye', handeye_result)
        
        # ★ 使用完整的 XYZ（不再丢弃 Z）
        fine_pos = handeye_result['position']
        fine_yaw = handeye_result['yaw']
        
        # Apply offset
        if offset_xy is None:
            offset_xy = [0.0, 0.0]
        
        final_position = np.array([
            fine_pos[0] + offset_xy[0],
            fine_pos[1] + offset_xy[1],
            fine_pos[2],  # ★ 使用手眼相机的真实 Z
        ])
        
        if log_step:
            print(f"  Hand-eye XYZ: [{fine_pos[0]:.4f}, {fine_pos[1]:.4f}, {fine_pos[2]:.4f}] m")
            print(f"  Head ref  Z:  {reference_z:.4f} m  →  Hand-eye Z: {fine_pos[2]:.4f} m  "
                f"(delta={fine_pos[2]-reference_z:.4f})")
            if abs(offset_xy[0]) > 0.001 or abs(offset_xy[1]) > 0.001:
                print(f"  Offset: [{offset_xy[0]:.4f}, {offset_xy[1]:.4f}]")
            print(f"  Final: [{final_position[0]:.4f}, {final_position[1]:.4f}, {final_position[2]:.4f}] m")
        
        # Move to aligned position — XY alignment, keep current Z (still at hover height)
        current_pos = self.robot_env.get_tcp_position()
        if current_pos is None:
            return False, None, "Cannot get TCP position"
        
        align_pos = [final_position[0], final_position[1], current_pos[2]]

        # ★ 用手眼的 XY + YAW 对齐：在悬停高度先把夹爪绕Z转到 fine_yaw（不再沿用头部朝向）。
        #   move_arm 内部 _compute_grasp_orientation 已含 ±180° 对称规整，不会顶腕关节极限。
        if not self.robot_env.move_arm(align_pos, fine_yaw, wait=1.5, check_abort=self._check_abort):
            return False, None, "XY+Yaw alignment failed"

        if log_step:
            print(f"  XY+Yaw aligned (yaw={np.degrees(fine_yaw):.1f}°)")
        
        result = {
            'position': final_position,  # ★ 完整 XYZ
            'brick_center': [fine_pos[0], fine_pos[1]],
            'yaw': fine_yaw,
            'offset_applied': offset_xy,
            'z_compensation': z_compensation,
            'refined_z': float(fine_pos[2]),  # ★ 手眼校准后的 Z
        }
        
        self._context['fine_position'] = final_position.tolist()
        self._context['fine_yaw'] = float(fine_yaw)
        self._context['fine_z'] = float(fine_pos[2])  # ★ 存储校准后的 Z
        self._context['brick_center'] = [fine_pos[0], fine_pos[1]]
        
        return True, result, None
    
    # ==================== Atomic Action 4: Open Gripper ====================
    
    def open_gripper(
        self,
        gap: float = 0.08,
        log_step: bool = True,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        [Atomic] Open gripper to specified gap.
        
        Args:
            gap: Target gripper opening (meters)
            log_step: Whether to log step message
            
        Returns:
            (success, result_dict, error_message)
        """
        if log_step:
            self._log_step(f"\n[Atomic: open_gripper] Opening to {gap*1000:.1f} mm...")
        
        if self._check_abort():
            return False, None, "Aborted"
        
        if not self.robot_env.open_gripper(gap):
            return False, None, "Open gripper failed"
        
        self._context['gripper_gap'] = gap
        
        if log_step:
            print(f"  Gripper opened")
        
        return True, {'gap': gap}, None
    
    # ==================== Atomic Action 5: Descend to Z ====================
    
    def descend_to_z(
        self,
        target_z: float,
        target_yaw: float,
        wait: float = 2.0,
        log_step: bool = True,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        [Atomic] Descend arm to specified Z height (keep XY).
        
        Args:
            target_z: Target Z height
            target_yaw: Yaw angle to maintain
            wait: Wait time after move
            log_step: Whether to log step message
            
        Returns:
            (success, result_dict, error_message)
        """
        if log_step:
            self._log_step(f"\n[Atomic: descend_to_z] Descending to Z={target_z:.4f} m...")
        
        if self._check_abort():
            return False, None, "Aborted"
        
        tcp_pos = self.robot_env.get_tcp_position()
        if tcp_pos is None:
            return False, None, "Cannot get TCP position"
        
        descend_pos = [tcp_pos[0], tcp_pos[1], target_z]
        
        if log_step:
            print(f"  From Z={tcp_pos[2]:.4f} m, descending {(tcp_pos[2] - target_z)*1000:.1f} mm")
        
        if not self.robot_env.move_arm(descend_pos, target_yaw, wait=wait, check_abort=self._check_abort,
                                       **_tilt(pitch, roll)):
            return False, None, "Descend failed"
        
        self._context['descend_position'] = descend_pos
        
        if log_step:
            print(f"  Descended")
        
        return True, {'position': descend_pos, 'yaw': target_yaw}, None
    
    # ==================== Atomic Action 6: Close Gripper ====================
    
    def close_gripper(
        self,
        check_grasp: bool = True,
        effort_threshold: float = 2.0, # bricks 2.0 lego 0.2
        min_gap_ratio: float = 0.5,
        brick_width: float = 0.05, # ricks 0.05 lego 0.015
        log_step: bool = True,
        close_step: Optional[float] = None,
        grip_rise: Optional[float] = None,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        [Atomic] Close gripper with optional grasp check.
        
        Args:
            check_grasp: Whether to verify grasp success
            effort_threshold: Minimum effort for successful grasp
            min_gap_ratio: Minimum gap as ratio of brick width
            brick_width: Expected brick width (meters)
            log_step: Whether to log step message
            
        Returns:
            (success, result_dict, error_message)
        """
        if log_step:
            self._log_step("\n[Atomic: close_gripper] Closing gripper...")
        
        if self._check_abort():
            return False, None, "Aborted"

        # close_step opts into closing in increments and stopping the moment the fingers stop moving.
        # Left as None the gripper closes fully in one command.
        gradual = None
        if close_step and hasattr(self.robot_env, "close_gripper_gradually"):
            kw = {} if grip_rise is None else {"effort_rise": float(grip_rise)}
            gradual = self.robot_env.close_gripper_gradually(step=float(close_step),
                                                             check_abort=self._check_abort, **kw)
            if not gradual.get("ok"):
                return False, gradual, gradual.get("note") or "Close gripper failed"
            if log_step:
                print(f"  {gradual['note']} ({gradual['steps']} steps)")
        elif not self.robot_env.close_gripper():
            return False, None, "Close gripper failed"

        time.sleep(0.3)
        
        gripper_state = self.robot_env.get_gripper_state()
        effort = gripper_state['effort'] or 0.0
        gap = gripper_state['gap'] or 0.0
        
        if log_step:
            print(f"  Effort: {effort:.3f} A, Gap: {gap:.4f} m")
        
        result = {'effort': effort, 'gap': gap}

        if gradual is not None:
            # Everything the closure measured, not a summary of it. The first version forwarded three
            # keys and dropped the effort trace, so the numbers that decide a grasp never reached the
            # caller; they survived only because they happened to be quoted inside `note`.
            result.update({k: v for k, v in gradual.items() if k not in ('ok', 'held')})
            result.update({'closed_by': 'increments', 'stopped_early': gradual['held']})
            if check_grasp and gradual['held'] is not None:
                # Where the closure stopped beats a fixed minimum gap: it has no thickness floor, so
                # it works on a sheet of cloth as well as on a brick.
                if not gradual['held']:
                    return False, result, ("Grasp not detected: the fingers closed the whole way, so "
                                           "nothing is between them")
                self._context['gripper_effort'] = effort
                self._context['gripper_gap'] = gap
                return True, result, None

        if check_grasp:
            # gap-based detection: gap > min means something is between fingers
            # gap < max means fingers actually moved (not stuck open)
            min_gap = 0.005   # 5mm: not fully closed → something held
            max_gap = 0.07    # 70mm: not still fully open
            # This branch has never once been taken: gripper effort was read from a dictionary key
            # that did not exist, so `effort` was always 0.0 and every result came from the gap. Now
            # that the effort is real, taking the branch would silently change results that were
            # measured without it — a 22.9 mm brick grasp reads as held on the gap and NOT held on
            # `gap > brick_width * min_gap_ratio`. So the legacy path stays where it has always
            # actually been, and the threshold is left uncalibrated rather than pretending it was.
            # The harness opts into the better test through close_step instead.
            if False and effort > effort_threshold:
                grasp_detected = (gap > brick_width * min_gap_ratio)
            else:
                grasp_detected = (min_gap < gap < max_gap)
            
            if not grasp_detected:
                return False, result, "Grasp not detected"
        
        self._context['gripper_effort'] = effort
        self._context['gripper_gap'] = gap
        
        return True, result, None
    
    # ==================== Atomic Action 7: Lift By ====================
    
    def lift_by(
        self,
        height: float = 0.10,
        target_yaw: Optional[float] = None,
        wait: float = 2.0,
        log_step: bool = True,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        [Atomic] Lift arm by specified height.
        
        Args:
            height: Height to lift (meters)
            target_yaw: Yaw angle (if None, uses context)
            wait: Wait time after move
            log_step: Whether to log step message
            
        Returns:
            (success, result_dict, error_message)
        """
        if log_step:
            self._log_step(f"\n[Atomic: lift_by] Lifting {height*1000:.1f} mm...")
        
        if self._check_abort():
            return False, None, "Aborted"
        
        tcp_pos = self.robot_env.get_tcp_position()
        if tcp_pos is None:
            return False, None, "Cannot get TCP position"
        tcp_pos_list = tcp_pos.tolist()
        
        if target_yaw is None:
            target_yaw = self._context.get('fine_yaw', 0.0)
        
        lift_pos = [tcp_pos_list[0], tcp_pos_list[1], tcp_pos_list[2] + height]
        
        if log_step:
            print(f"  From Z={tcp_pos_list[2]:.4f} to Z={lift_pos[2]:.4f} m")
        
        # pitch and roll default to 0 (straight down).
        # The harness passes the angles the tool is ALREADY holding: changing height is not a reason to
        # change orientation, and doing it silently cost a cloth grasp — the fingers pinched a fold at
        # 35 degrees and the lift rotated them upright before anything could be carried anywhere.
        if not self.robot_env.move_arm(lift_pos, target_yaw, wait=wait, check_abort=self._check_abort,
                                       pitch=pitch, roll=roll):
            return False, None, "Lift failed"
        
        self._context['lift_position'] = lift_pos
        
        if log_step:
            print(f"  Lifted")
        
        return True, {'position': lift_pos, 'lift_height': height}, None
    
    # ==================== Simple Release (no LLM planner) ====================
    
    def _release_simple(
        self,
        target_surface_z: float,
        target_yaw: float,
        log_step: bool = True,
        effort_threshold: float = 13.0,
        descend_step: float = 0.005,
        max_depth: float = 0.015,
        retreat_height: float = 0.10,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Simple release: descend with current protection → open gripper → retreat."""
        tcp_pos = self.robot_env.get_tcp_position()
        if tcp_pos is None:
            return False, None, "Cannot get TCP position"

        current_pos = [float(tcp_pos[0]), float(tcp_pos[1]), target_surface_z]
        if log_step:
            print(f"  Descending to Z={target_surface_z:.4f} m")

        if not self.robot_env.move_arm(current_pos, target_yaw, wait=2.0,
                                       check_abort=self._check_abort):
            return False, None, "Place descend failed"

        # Press down with current protection
        total_pressed = 0.0
        contact = False

        while total_pressed < max_depth:
            if self._check_abort():
                return False, None, "Aborted"

            arm_effort = self.robot_env.get_arm_total_effort()
            if arm_effort is not None and arm_effort > effort_threshold:
                contact = True
                if log_step:
                    print(f"  Contact: {arm_effort:.2f}A > {effort_threshold:.1f}A")
                break

            current_pos[2] -= descend_step
            total_pressed += descend_step
            self.robot_env.move_arm(current_pos, target_yaw, wait=0.1,
                                    check_abort=self._check_abort)
            time.sleep(0.02)

        # Open gripper
        if not self.robot_env.open_gripper(0.08):
            return False, None, "Gripper open failed"
        time.sleep(0.3)

        # Retreat
        retreat_pos = [current_pos[0], current_pos[1], current_pos[2] + retreat_height]
        self.robot_env.move_arm(retreat_pos, target_yaw, wait=1.5,
                                check_abort=self._check_abort)

        if log_step:
            status = "contact" if contact else "max_depth"
            print(f"  Released ({status}, pressed {total_pressed*1000:.1f}mm)")

        return True, {
            'final_place_position': current_pos,
            'contact_detected': contact,
            'total_pressed': total_pressed,
        }, None
        
    # ==================== Atomic Action 8: Release with Contact ====================
    
    def release_with_contact(
        self,
        target_surface_z: float,
        target_yaw: float,
        log_step: bool = True,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        [Atomic] Descend to surface with contact detection and release.
        
        Args:
            target_surface_z: Target Z for placement
            target_yaw: Yaw angle for placement
            log_step: Whether to log step message
            
        Returns:
            (success, result_dict, error_message)
        """
        if log_step:
            self._log_step("\n[Atomic: release_with_contact] Contact release...")
        
        if self._check_abort():
            return False, None, "Aborted"

        if self.llm_planner is None:
            return self._release_simple(target_surface_z, target_yaw, log_step,
                                        max_depth=self.release_max_depth)
        
        tcp_pos = self.robot_env.get_tcp_position()
        if tcp_pos is None:
            return False, None, "Cannot get TCP position"
        tcp_pos_list = tcp_pos.tolist()
        
        final_place_pos = [tcp_pos_list[0], tcp_pos_list[1], target_surface_z]
        
        if log_step:
            print(f"  Target Z: {target_surface_z:.4f} m")
        
        if not self.robot_env.move_arm(final_place_pos, target_yaw, wait=2.5, check_abort=self._check_abort):
            return False, None, "Place descend failed"
        
        # Get release config
        release_config = self.llm_planner.config.get("release", {})
        max_attempts = release_config.get("max_attempts", 10)
        contact_threshold_mm = release_config.get("contact_threshold_mm", 0.8)
        descend_step = release_config.get("descend_step", 0.005)
        lift_step = release_config.get("lift_step", 0.01)
        arm_effort_threshold = release_config.get("arm_effort_threshold", 12.0)
        
        current_place_pos = list(final_place_pos)
        
        for attempt in range(1, max_attempts + 1):
            if self._check_abort():
                return False, None, "Aborted"
            
            tcp_pos = self.robot_env.get_tcp_position()
            if tcp_pos is None:
                continue
            
            actual_z = tcp_pos[2]
            target_z = current_place_pos[2]
            arm_effort = self.robot_env.get_arm_total_effort()
            
            # Real-time effort protection
            if arm_effort is not None and arm_effort > arm_effort_threshold:
                if log_step:
                    print(f"  ⚠️ Effort protection! Lifting...")
                current_place_pos[2] += lift_step
                self.robot_env.move_arm(current_place_pos, target_yaw, wait=0.5, check_abort=self._check_abort)
                break
            
            # LLM analysis
            success, release_result, error = self.llm_planner.analyze_release_feedback(
                actual_z=actual_z,
                target_z=target_z,
                contact_threshold_mm=contact_threshold_mm,
                descend_step=descend_step,
                lift_step=lift_step,
                attempt_number=attempt,
                max_attempts=max_attempts,
                arm_effort=arm_effort,
                arm_effort_threshold=arm_effort_threshold,
            )
            
            if not success:
                current_place_pos[2] -= descend_step
                self.robot_env.move_arm(current_place_pos, target_yaw, wait=1.0, check_abort=self._check_abort)
                continue
            
            action = release_result['action']
            action_type = action['type']
            delta_z = action['delta_z']
            
            if action_type == 'release':
                if log_step:
                    print("  ✓ Contact detected, releasing!")
                break
            elif action_type == 'lift_then_release':
                if log_step:
                    print(f"  Lifting {delta_z*1000:.1f} mm before release...")
                current_place_pos[2] += delta_z
                self.robot_env.move_arm(current_place_pos, target_yaw, wait=1.0, check_abort=self._check_abort)
                break
            elif action_type == 'descend':
                current_place_pos[2] += delta_z
                self.robot_env.move_arm(current_place_pos, target_yaw, wait=1.0, check_abort=self._check_abort)
        
        # Release gripper
        if not self.robot_env.open_gripper(0.08):
            return False, None, "Gripper open failed"
        
        time.sleep(0.3)
        
        # Retreat
        retreat_pos = [current_place_pos[0], current_place_pos[1], current_place_pos[2] + 0.10]
        self.robot_env.move_arm(retreat_pos, target_yaw, wait=1.5, check_abort=self._check_abort)
        
        if log_step:
            print("  ✓ Released!")
        
        return True, {'final_place_position': current_place_pos}, None


def create_atomic_executor(
    robot_env,
    segmenter,
    head_calc,
    handeye_calc,
    llm_planner=None,
    prompt: str = "block, brick, rectangular object",
) -> AtomicMotionExecutor:
    """Factory function to create atomic executor."""
    return AtomicMotionExecutor(
        robot_env=robot_env,
        segmenter=segmenter,
        head_calc=head_calc,
        handeye_calc=handeye_calc,
        llm_planner=llm_planner,
        prompt=prompt,
    )