from __future__ import annotations

import torch
from lightning.pytorch.callbacks.progress import TQDMProgressBar


class FourDecimalProgressBar(TQDMProgressBar):
    """tqdm 진행률 막대에 표시되는 부동소수 값을 소수점 넷째 자리까지 고정 출력한다."""

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        formatted: dict[str, str] = {}

        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                # 대표 스칼라 텐서만 추출해 문자열 포맷 적용
                if value.numel() == 1:
                    value = value.item()
                else:
                    formatted[key] = str(value)
                    continue

            if isinstance(value, float):
                formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = value

        return formatted
