"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes


def summarize_per_class_ap(coco_eval, class_names=None):
    """
    COCOeval 결과에서 클래스별 AP50, AP75, mAP50-95를 계산합니다.

    반환 예:
    {
        "1": {
            "AP50": 0.9123,
            "AP75": 0.7345,
            "mAP50-95": 0.6234
        },
        ...
    }
    """

    precisions = coco_eval.eval["precision"]
    # precision shape: [T, R, K, A, M]
    # T: IoU thresholds, 0.50:0.95
    # R: recall thresholds
    # K: classes
    # A: area range
    # M: max detections

    cat_ids = coco_eval.params.catIds
    iou_thrs = coco_eval.params.iouThrs

    results = {}

    for class_idx, cat_id in enumerate(cat_ids):
        class_result = {}

        # mAP50-95: IoU 0.50~0.95 전체 평균
        precision_all = precisions[:, :, class_idx, 0, -1]
        precision_all = precision_all[precision_all > -1]
        class_result["mAP50-95"] = (
            float(np.mean(precision_all)) if precision_all.size else float("nan")
        )

        # AP50
        iou_50_idx = np.where(np.isclose(iou_thrs, 0.50))[0]
        if len(iou_50_idx) > 0:
            precision_50 = precisions[iou_50_idx[0], :, class_idx, 0, -1]
            precision_50 = precision_50[precision_50 > -1]
            class_result["AP50"] = (
                float(np.mean(precision_50)) if precision_50.size else float("nan")
            )
        else:
            class_result["AP50"] = float("nan")

        # AP75
        iou_75_idx = np.where(np.isclose(iou_thrs, 0.75))[0]
        if len(iou_75_idx) > 0:
            precision_75 = precisions[iou_75_idx[0], :, class_idx, 0, -1]
            precision_75 = precision_75[precision_75 > -1]
            class_result["AP75"] = (
                float(np.mean(precision_75)) if precision_75.size else float("nan")
            )
        else:
            class_result["AP75"] = float("nan")

        if class_names is not None:
            class_name = class_names.get(cat_id, str(cat_id))
        else:
            class_name = str(cat_id)

        results[class_name] = class_result

    return results


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = (
        "Epoch: [{}]".format(epoch)
        if epochs is None
        else "Epoch: [{}/{}]".format(epoch, epochs)
    )

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    accumulation_steps = 8  # 배치사이즈 설정값 x 8로 효과보도록
    optimizer.zero_grad()

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(
            epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader)
        )

        if (
            global_step < num_visualization_sample_batch
            and output_dir is not None
            and dist_utils.is_main_process()
        ):
            save_samples(
                samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh"
            )

        samples = samples.to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        # 배치가 다 차기 전이나 마지막 자투리 배치일 때 업데이트 여부 판단 조건
        is_update_step = (i + 1) % accumulation_steps == 0 or (i + 1) == len(
            data_loader
        )
        # ----------------------------------------------------

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if (
                torch.isnan(outputs["pred_boxes"]).any()
                or torch.isinf(outputs["pred_boxes"]).any()
            ):
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())

            loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if is_update_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            loss = loss / accumulation_steps
            loss.backward()

            if is_update_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()
                optimizer.zero_grad()

        # ema
        if is_update_step:
            if ema is not None:
                ema.update(model)

            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        global_step = epoch * len(data_loader) + i

        if (
            global_step < num_visualization_sample_batch
            and output_dir is not None
            and dist_utils.is_main_process()
        ):
            save_samples(
                samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy"
            )

        samples = samples.to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        outputs = model(samples)
        try:
            metas = dict(
                epoch=epoch,
                step=i,
                global_step=epoch * len(data_loader) + i,
                epoch_step=len(data_loader),
            )
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
            val_loss_value = sum(loss_dict.values()).item()
            metric_logger.update(val_loss=val_loss_value)
        except Exception as e:
            # loss 계산 실패해도 나머지 평가는 계속 진행
            pass

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # validator format for metrics
        for idx, (target, result) in enumerate(zip(targets, results)):
            gt.append(
                {
                    "boxes": scale_boxes(  # from model input size to original img size
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": target["labels"],
                }
            )
            labels = (
                (
                    torch.tensor(
                        [
                            mscoco_category2label[int(x.item())]
                            for x in result["labels"].flatten()
                        ]
                    )
                    .to(result["labels"].device)
                    .reshape(result["labels"].shape)
                )
                if postprocessor.remap_mscoco_category
                else result["labels"]
            )
            preds.append(
                {"boxes": result["boxes"], "labels": labels, "scores": result["scores"]}
            )

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}

    # val_loss 등 metric_logger에 있는 값 저장
    stats.update({k: meter.global_avg for k, meter in metric_logger.meters.items()})

    if coco_evaluator is not None:
        if "bbox" in iou_types:
            bbox_eval = coco_evaluator.coco_eval["bbox"]
            coco_stats = bbox_eval.stats.tolist()

            # COCO 전체 지표
            stats["coco_eval_bbox"] = coco_stats

            # 전체 bbox 지표
            stats["mAP50-95"] = coco_stats[0]  # AP @[ IoU=0.50:0.95 ]
            stats["AP50"] = coco_stats[1]  # AP @[ IoU=0.50 ]
            stats["AP75"] = coco_stats[2]  # AP @[ IoU=0.75 ]

            # 클래스별 지표
            per_class_ap = summarize_per_class_ap(bbox_eval)
            stats["per_class_ap"] = per_class_ap

            if dist_utils.is_main_process():
                print("\nPer-class bbox AP")
                print("-" * 70)
                print(f"{'Class':<20} {'AP50':>10} {'AP75':>10} {'mAP50-95':>12}")
                print("-" * 70)

                for class_name, values in per_class_ap.items():
                    print(
                        f"{class_name:<20} "
                        f"{values['AP50']:>10.4f} "
                        f"{values['AP75']:>10.4f} "
                        f"{values['mAP50-95']:>12.4f}"
                    )

                print("-" * 70)

    return stats, coco_evaluator

    # stats = {}
    # # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if "bbox" in iou_types:
    #         stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

    # return stats, coco_evaluator
