"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb,
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            # ----- 수정 코드 -----
            if self.output_dir:  # 매 에폭 저장
                checkpoint_paths = [self.output_dir / "last.pth"]

                # 사용자가 지정한 가중치 저장 주기(args.checkpoint_freq)마다 백업용 스냅샷을 영구 저장합니다.
                # if (epoch + 1) % args.checkpoint_freq == 0:
                if (epoch + 1) % 5 == 0:  # 2회마다 저장
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch:04}.pth"
                    )

                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # ----- 기존 코드 -----
            # if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
            #     checkpoint_paths = [self.output_dir / "last.pth"]
            #     # extra checkpoint before LR drop and every 100 epochs
            #     if (epoch + 1) % args.checkpoint_freq == 0:
            #         checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
            #     for checkpoint_path in checkpoint_paths:
            #         dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            # TODO
            # TensorBoard logging
            for k, value in test_stats.items():
                if self.writer and dist_utils.is_main_process():

                    # 기존 coco_eval_bbox, coco_eval_masks 같은 list/tuple/array 값
                    if isinstance(value, (list, tuple)):
                        for i, v in enumerate(value):
                            self.writer.add_scalar(f"Test/{k}_{i}", float(v), epoch)

                    # AP50, AP75, mAP50-95 같은 float 값
                    elif isinstance(value, (int, float)):
                        self.writer.add_scalar(f"Test/{k}", float(value), epoch)

                    # per_class_ap 같은 dict 값
                    elif isinstance(value, dict):
                        for cls_name, ap_dict in value.items():
                            if isinstance(ap_dict, dict):
                                for metric_name, metric_value in ap_dict.items():
                                    self.writer.add_scalar(
                                        f"Test/per_class/{cls_name}_{metric_name}",
                                        float(metric_value),
                                        epoch,
                                    )

            # Best checkpoint 기준 metric 선택
            if "coco_eval_bbox" in test_stats:
                current_metric = test_stats["coco_eval_bbox"][0]
            elif "mAP50-95" in test_stats:
                current_metric = test_stats["mAP50-95"]
            else:
                current_metric = None

            if current_metric is not None:
                current_metric = float(current_metric)

                if "coco_eval_bbox" in best_stat:
                    best_stat["epoch"] = (
                        epoch
                        if current_metric > best_stat["coco_eval_bbox"]
                        else best_stat["epoch"]
                    )
                    best_stat["coco_eval_bbox"] = max(
                        best_stat["coco_eval_bbox"], current_metric
                    )
                else:
                    best_stat["epoch"] = epoch
                    best_stat["coco_eval_bbox"] = current_metric

                if best_stat["coco_eval_bbox"] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat["coco_eval_bbox"]

                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                        else:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                best_stat_print["coco_eval_bbox"] = max(
                    best_stat["coco_eval_bbox"], top1
                )
                print(f"best_stat: {best_stat_print}")  # global best

                # if best_stat["epoch"] == epoch and self.output_dir:
                #     if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                #         if test_stats[k][0] > top1:
                #             top1 = test_stats[k][0]
                #             dist_utils.save_on_master(
                #                 self.state_dict(), self.output_dir / "best_stg2.pth"
                #             )
                #     else:
                #         top1 = max(test_stats[k][0], top1)
                #         dist_utils.save_on_master(
                #             self.state_dict(), self.output_dir / "best_stg1.pth"
                #         )

                # elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {
                        "epoch": -1,
                    }
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        print(
                            f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}"
                        )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        _, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
