# coding=utf-8
# @Time    : 2021/3/4
# @Author  : ZTY

from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_decoder import TransformerDecoder, TransformerDecoderBase, Linear
from .futurecoder import TransformerFuturecoder, TransformerFuturecoderBase
from .fc_transformer_model import (
    FutureContextTransformerModel,
    base_architecture,
)
from .fc_transformer_base import FutureContextTransformerModelBase
from .mm_translation import MultimodalTranslationTask1
from .fc_label_smoothed_cross_entropy import FutureContextLabelSmoothedCrossEntropyCriterion


__all__ = [
    "FutureContextTransformerModelBase",
    "TransformerConfig",
    "TransformerDecoder",
    "TransformerDecoderBase",
    "TransformerFuturecoder",
    "TransformerFuturecoderBase",
    "FutureContextTransformerModel",
    "Linear",
    "base_architecture",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
    "MultimodalTranslationTask1"
]