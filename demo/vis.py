#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import random
import logging
import numpy as np
import cv2
import math
from typing import List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualizer:
    """A visualizer for various Computer Vision tasks based on MLLM text outputs."""

    def __init__(self):
        self.color_palette = self._generate_color_palette()
        # Track label positions to avoid overlaps
        self.used_label_rects = []

    def visualize(self, img: np.ndarray, text: str) -> Optional[np.ndarray]:
        """Automatically dispatches the correct visualization method based on text content."""
        if img is None: return None
        
        self.used_label_rects = [] # Reset collision tracker
        orig_h, orig_w = img.shape[:2]
        
        try:
            if '<ins>' in text and '<ref>' not in text:
                result = self.vis_referring_seg(img, text)
            elif '<mask>' in text and '<depth>' in text:
                result = self.vis_depth(img, text)
            elif '<mask>' in text and '<ref>' in text:
                result = self.vis_semantic_seg(img, text)
            elif '<person>' in text or '<kpt>' in text:
                result = self.vis_pose(img, text)
            elif '<box>' in text:
                result = self.vis_box(img, text)
            else:
                return img
            
            if result is not None and result.shape[:2] != (orig_h, orig_w):
                result = cv2.resize(result, (orig_w, orig_h))
            return result
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return img
            
    def _generate_color_palette(self, n_colors: int = 256) -> List[Tuple[int, int, int]]:
        """Generates a diverse color palette using Golden Angle in HSV space."""
        palette = []
        for i in range(n_colors):
            h = (i * 137.508) % 360  # Golden angle distribution
            s = 0.7 + (i % 3) * 0.1
            v = 0.9 - (i % 4) * 0.1
            
            # Simple HSV to BGR conversion
            c = v * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = v - c
            if 0 <= h < 60:    r, g, b = c, x, 0
            elif 60 <= h < 120:  r, g, b = x, c, 0
            elif 120 <= h < 180: r, g, b = 0, c, x
            elif 180 <= h < 240: r, g, b = 0, x, c
            elif 240 <= h < 300: r, g, b = x, 0, c
            else:               r, g, b = c, 0, x
            
            palette.append((int((b + m) * 255), int((g + m) * 255), int((r + m) * 255)))
        return palette

    def _get_random_color(self) -> Tuple[int, int, int]:
        return tuple(random.choices(range(50, 256), k=3))

    def _is_overlapping(self, new_rect: Tuple[int, int, int, int], threshold: float = 0.1) -> bool:
        """Checks if a new label rectangle overlaps significantly with existing labels."""
        nx1, ny1, nx2, ny2 = new_rect
        for (ox1, oy1, ox2, oy2) in self.used_label_rects:
            # Calculate intersection
            ix1 = max(nx1, ox1)
            iy1 = max(ny1, oy1)
            ix2 = min(nx2, ox2)
            iy2 = min(ny2, oy2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                new_area = (nx2 - nx1) * (ny2 - ny1)
                if intersection_area / new_area > threshold:
                    return True
        return False

    def _draw_label(self, img: np.ndarray, text: str, position: Tuple[int, int], 
                   color: Tuple[int, int, int], bg: bool = True):
        """Draws a label text with background, avoiding collisions with previous labels."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Try multiple positions near the target to avoid overlap
        best_pos = None
        offsets = [(0, 0), (0, -th - 10), (0, th + 10), (tw + 10, 0), (-tw - 10, 0)]
        
        for dx, dy in offsets:
            x, y = position[0] + dx, position[1] + dy
            x = max(2, min(x, img.shape[1] - tw - 2))
            y = max(th + 2, min(y, img.shape[0] - baseline - 2))
            rect = (x - 2, y - th - 2, x + tw + 2, y + baseline + 2)
            
            if not self._is_overlapping(rect):
                best_pos = (x, y, rect)
                break
        
        # If all overlap, just use the last calculated one but it's rare with offsets
        if best_pos is None:
            best_pos = (x, y, rect)

        final_x, final_y, final_rect = best_pos
        self.used_label_rects.append(final_rect)

        if bg:
            cv2.rectangle(img, (final_rect[0], final_rect[1]), (final_rect[2], final_rect[3]), color, -1)
            # Contrast-aware text color
            luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
            text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
        else:
            text_color = (255, 255, 255)
        
        cv2.putText(img, text, (final_x, final_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


    def _parse_semantic_seg(self, text: str) -> Tuple[List[str], Optional[np.ndarray]]:
        classes = re.findall(r'<ref>(.*?)</ref>', text)
        mask_match = re.search(r'<mask>(.*?)</mask>', text)
        if not mask_match: return classes, None
        indices = [int(i) for i in mask_match.group(1).split(',') if i.strip()]
        return classes, np.array(indices, dtype=np.int32)
        
    def vis_semantic_seg(self, img: np.ndarray, text: str) -> np.ndarray:
        """Visualizes semantic segmentation with robust mask reshaping."""
        h, w = img.shape[:2]
        class_names, pred_mask_1d = self._parse_semantic_seg(text)
        
        if pred_mask_1d is None: return img
        
        mask_len = pred_mask_1d.shape[0]
        pred_mask_2d = None

        # Logic to find correct dimensions
        if mask_len == h * w:
            pred_mask_2d = pred_mask_1d.reshape((h, w))
        else:
            return img

        if pred_mask_2d is None: return img

        vis_map = img.copy()
        unique_labels = np.unique(pred_mask_2d)
        
        for lb in unique_labels:
            if lb == 0 and len(unique_labels) > 1: continue # Usually background
            color = self.color_palette[lb % len(self.color_palette)]
            mask = (pred_mask_2d == lb)
            
            # Blend color
            vis_map[mask] = cv2.addWeighted(vis_map[mask], 0.5, 
                                           np.full_like(vis_map[mask], color), 0.5, 0)
            
            # Label placement
            coords = np.argwhere(mask)
            if coords.size > 0 and lb < len(class_names):
                ymin, xmin = coords.min(axis=0)
                cx = int(xmin + 20)
                cy = int(ymin + 20)
                self._draw_label(vis_map, class_names[lb], (cx, cy), color)
        
        return vis_map

    def vis_box(self, img: np.ndarray, text: str) -> np.ndarray:
            """Visualizes object detection bounding boxes (supports no-label boxes)."""
            vis_img = img.copy()
            
            # extract labels and boxes
            tags = list(re.finditer(r"<(?:label|ref)>(.*?)</(?:label|ref)>", text))
            box_pattern = r"<box>\s*<x_([\d\.]+)>\s*<y_([\d\.]+)>\s*<x_([\d\.]+)>\s*<y_([\d\.]+)>\s*</box>"
            boxes = list(re.finditer(box_pattern, text))
            
            for box_m in boxes:
                prev_tags = [t.group(1) for t in tags if t.end() <= box_m.start()]
                label = prev_tags[-1] if prev_tags else "" 
                
                color = self.color_palette[hash(label) % len(self.color_palette)]
                coords = [float(box_m.group(j)) for j in range(1, 5)]
                x1, y1, x2, y2 = map(int, coords)
                
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                if label:
                    self._draw_label(vis_img, label, (x1, y1 - 5), color)
            
            return vis_img

    def _parse_kpts_mpii_multi(self, text: str):
        persons = re.findall(r"<person>(.*?)</person>", text, re.DOTALL)
        all_person_kpts = []
        for p in persons:
            kpt_blocks = re.findall(r"<kpt>(.*?)</kpt>", p, re.DOTALL)
            kpts = []
            for k in kpt_blocks:
                x = y = v = math.nan
                m = re.search(r"<x_([^>]+)>", k)
                if m: x = float(m.group(1))
                m = re.search(r"<y_([^>]+)>", k)
                if m: y = float(m.group(1))
                m = re.search(r"<v_([^>]+)>", k)
                if m: v = float(m.group(1))
                kpts.append((x, y, v))
            if len(kpts) >= 16:
                all_person_kpts.append(kpts)
        return all_person_kpts

    def vis_pose(self, img: np.ndarray, text: str) -> np.ndarray:
        """Visualize pose in MPII keypoints"""
        v_thresh = 0.5
        thickness = 5
        point_radius = 10
        draw_index = False

        # ===== MPII skeleton =====
        LEFT = [(5, 4), (4, 3), (3, 6), (13, 14), (14, 15)]
        RIGHT = [(0, 1), (1, 2), (2, 6), (12, 11), (11, 10)]
        CENTER = [(6, 7), (7, 8), (8, 9), (13, 12)]

        COLORS = {
            "left":   (102, 204, 255),
            "right":  (255, 153, 102),
            "center": (160, 160, 160),
        }

        POINT_VISIBLE   = (0, 0, 255)
        POINT_INVISIBLE = (255, 0, 0) 

        vis_img = img.copy()
        all_persons = self._parse_kpts_mpii_multi(text)
        h, w = vis_img.shape[:2]

        def valid(p):
            x, y, _ = p
            return (
                not math.isnan(x) and not math.isnan(y)
                and 0 <= x < w and 0 <= y < h
            )

        skeleton_groups = {"left": LEFT, "right": RIGHT, "center": CENTER}

        for pid, kpts in enumerate(all_persons):
            brightness = min(1.0, 0.7 + 0.3 * pid)

            def scale_color(c):
                return tuple(int(x * brightness) for x in c)

            for part, connections in skeleton_groups.items():
                color = scale_color(COLORS[part])

                for i, j in connections:
                    if i >= len(kpts) or j >= len(kpts):
                        continue

                    p1 = kpts[i]
                    p2 = kpts[j]

                    if valid(p1) and valid(p2):
                        x1, y1, _ = p1
                        x2, y2, _ = p2

                        cv2.line(
                            vis_img,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color,
                            thickness,
                            cv2.LINE_AA,
                        )

            for idx, (x, y, v) in enumerate(kpts):
                if not valid((x, y, v)):
                    continue

                center = (int(x), int(y))

                if v >= v_thresh:
                    cv2.circle(
                        vis_img,
                        center,
                        point_radius, 
                        POINT_VISIBLE,
                        -1,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.circle(
                        vis_img,
                        center,
                        point_radius, 
                        POINT_INVISIBLE,
                        -1,
                        cv2.LINE_AA,
                    )

                if draw_index:
                    cv2.putText(
                        vis_img,
                        str(idx),
                        (center[0] + 3, center[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            head_idx_candidates = [9, 8]  # HeadTop -> UpperNeck
            for hid in head_idx_candidates:
                if hid < len(kpts) and valid(kpts[hid]) and kpts[hid][2] >= v_thresh:
                    hx, hy, _ = kpts[hid]
                    cv2.putText(
                        vis_img,
                        f"P{pid}",
                        (int(hx), int(hy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        scale_color((255, 255, 255)),
                        2,
                        cv2.LINE_AA,
                    )
                    break

        return vis_img
        
    def vis_referring_seg(self, img: np.ndarray, text: str) -> np.ndarray:
        """Visualizes referring expression segmentation (single object poly)."""
        match = re.search(r"<poly>(.*?)</poly>", text)
        if not match: return img
        
        coords = re.findall(r"<(?:x|y)_([\d\.]+)>", match.group(1))
        pts = np.array([float(c) for c in coords]).reshape(-1, 2).astype(np.int32)
        
        if len(pts) < 3: return img
        
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (255, 0, 0))
        return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    def _parse_depth(self, text: str) -> Tuple[List[str], Optional[np.ndarray]]:
        mask_match = re.search(r'<mask>(.*?)</mask>', text)
        if not mask_match: return None
        indices = [int(i) for i in mask_match.group(1).split(',') if i.strip()]
        return np.array(indices, dtype=np.int32)
        
    def vis_depth(self, img: np.ndarray, text: str) -> np.ndarray:
        """Visualizes depth estimation maps."""
        match = re.search(r'<mask>(.*?)</mask>', text)
        if not match: return img

        h, w = img.shape[:2]
        pred_mask_1d = self._parse_depth(text)
        
        if pred_mask_1d is None: return img
        
        mask_len = pred_mask_1d.shape[0]
        pred_mask_2d = None

        # Logic to find correct dimensions
        if mask_len == h * w:
            pred_mask_2d = pred_mask_1d.reshape((h, w))
        else:
            return img

        if pred_mask_2d is None: return img
 
        # Normalize for visualization, the paper applies extra dequantization and color scheme
        depth_norm = cv2.normalize(pred_mask_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        return cv2.addWeighted(img, 0.05, depth_color, 0.95, 0)