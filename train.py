"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import torch

log_dir = "/opt/ml/model"
log_path = os.path.join(log_dir, "train.log")

os.makedirs(log_dir, exist_ok=True)


class Logger(object):
    def __init__(self, filename=log_path):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # 모니터 화면에 출력
        self.log.write(message)  # 로그 파일에 기록

    def flush(self):
        # 파이썬 3 프리프 등 호환성을 위해 필요
        self.terminal.flush()
        self.log.flush()


# print 출력을 Logger 클래스로 리다이렉트
sys.stdout = Logger(log_path)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS
from pprint import pprint

debug = False

if debug:

    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


class ConfigArgs:
    def __init__(self):
        # =================================================================
        # === 사용자 설정 구역: 옵션을 여기에 직접 입력하세요 ===
        # =================================================================
        self.config = "configs/dfine/dfine_hgnetv2_l_coco.yml"  # 사용할 yaml 설정 파일 경로 (필수)
        self.resume = (
            None  # 학습을 재개할 체크포인트 경로 (예: 'output/checkpoint.pth')
        )
        self.tuning = None  # 파인튜닝할 체크포인트 경로
        self.device = "cuda"
        self.seed = 42
        self.use_amp = True
        self.output_dir = "output/"
        self.summary_dir = None
        self.test_only = False  # 평가만 진행할지 여부 (True: 평가만, False: 학습 진행)

        # CLI 추가 업데이트 옵션 (없을 경우 빈 리스트)
        # 예: ['epochs=100', 'lr=0.0001'] 형태로 지정 가능
        self.update = []

        # 환경 설정
        self.print_method = "builtin"
        self.print_rank = 0
        self.local_rank = None
        # =================================================================


def main(args) -> None:
    """main"""
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scratch or resume or tuning at one time"

    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k not in ["update"] and v is not None
        }
    )

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if safe_get_rank() == 0:
        print("cfg: ")
        pprint(cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    # 파서 대신 설정 객체를 생성하여 바로 실행합니다.
    args = ConfigArgs()
    main(args)
