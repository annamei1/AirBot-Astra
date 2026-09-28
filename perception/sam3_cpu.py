# Portions adapted from SAM3: Copyright (c) Meta Platforms, Inc. and affiliates.
# All Rights Reserved. See third_party/SAM3_LICENSE.

"""CPU compatibility for the SAM3 image model.

Tested against SAM3 commit 2345a4ad109ac29c569da749c91d84f10dc08c40.
Builder overrides are temporary and serialized. Inference overrides are bound to
CPU model instances; torch.Tensor and upstream module methods are never replaced
process-wide during inference. CUDA models use the unmodified upstream code.
"""
import os
import threading
from types import MethodType
from unittest.mock import patch

import torch
import torchvision

_BUILD_LOCK = threading.Lock()


def select_device():
    device = os.environ.get("SAM3_DEVICE", "auto").lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("SAM3_DEVICE must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM3_DEVICE=cuda requested, but CUDA is unavailable")
    return device


def _cpu_mlp_forward(self, x):
    # The upstream fused activation casts to bfloat16 even with float32 CPU weights.
    x = self.act(self.fc1(x))
    return self.drop2(self.fc2(self.norm(self.drop1(x))))


def _cpu_encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
    """Adapted from SAM3 SequenceGeometryEncoder._encode_boxes (Meta, SAM License).

    Only the scale allocation differs: create it on the destination device,
    without pinning host memory for a CUDA transfer that does not exist on CPU.
    Keep projection, pooling, position encoding and label embeddings unchanged.
    """
    from sam3.model.box_ops import box_cxcywh_to_xyxy

    boxes_embed = None
    n_boxes, bs = boxes.shape[:2]
    if self.boxes_direct_project is not None:
        boxes_embed = self.boxes_direct_project(boxes)
    if self.boxes_pool_project is not None:
        h, w = img_feats.shape[-2:]
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = boxes_xyxy.new_tensor([w, h, w, h]).view(1, 1, 4)
        boxes_xyxy = boxes_xyxy * scale
        sampled = torchvision.ops.roi_align(
            img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size)
        assert list(sampled.shape) == [bs * n_boxes, self.d_model, self.roi_size, self.roi_size]
        proj = self.boxes_pool_project(sampled)
        proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
        boxes_embed = proj if boxes_embed is None else boxes_embed + proj
    if self.boxes_pos_enc_project is not None:
        cx, cy, w, h = boxes.unbind(-1)
        enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
        enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])
        proj = self.boxes_pos_enc_project(enc)
        boxes_embed = proj if boxes_embed is None else boxes_embed + proj
    return self.label_embed(boxes_labels.long()) + boxes_embed, boxes_mask


def build_image_model(checkpoint_path, device):
    import sam3.model_builder as builder
    from sam3.model.geometry_encoders import SequenceGeometryEncoder
    from sam3.model.vitdet import Mlp

    with _BUILD_LOCK:
        if device == "cuda":
            return builder.build_sam3_image_model(checkpoint_path=checkpoint_path, device=device)

        threads = int(os.environ.get("SAM3_CPU_THREADS", "4"))
        if threads < 1:
            raise ValueError("SAM3_CPU_THREADS must be a positive integer")
        # PyTorch's thread count is process-wide; avoid severe CPU oversubscription.
        torch.set_num_threads(threads)
        position_encoding = builder._create_position_encoding
        coords = builder.TransformerDecoder._get_coords
        # SAM3 hardcodes CUDA in these two eager caches. Lazy position encoding
        # already uses the input device; decoder cache coordinates must be on CPU.
        with patch.object(builder, "_create_position_encoding",
                          lambda precompute_resolution=None: position_encoding(None)), \
             patch.object(builder.TransformerDecoder, "_get_coords",
                          staticmethod(lambda h, w, device: coords(h, w, "cpu"))):
            model = builder.build_sam3_image_model(checkpoint_path=checkpoint_path, device="cpu")
        for module in model.modules():
            if isinstance(module, Mlp):
                module.forward = MethodType(_cpu_mlp_forward, module)
            elif isinstance(module, SequenceGeometryEncoder):
                module._encode_boxes = MethodType(_cpu_encode_boxes, module)
        return model
