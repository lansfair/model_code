"""Flexible patch embedding Module.

Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
by https://github.com/bwconrad/flexivit/
"""

import logging
from collections.abc import Iterable

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.data.constants import ModalitySpec

logger = logging.getLogger(__name__)


def _to_2tuple(x: int | tuple[int, ...]) -> tuple[int, int]:
    """Convert a scalar or 2-element iterable to a (h, w) tuple.
    
    将标量或二元可迭代对象转换为(h, w)元组格式。
    
    Args:
        x: 输入值，可以是整数或包含两个元素的元组
        
    Returns:
        长度为2的元组，表示高度和宽度
        
    Raises:
        TypeError: 当输入类型不符合要求时抛出
        AssertionError: 当元组长度不为2时抛出
    """
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, Iterable) and not isinstance(x, str):
        values = tuple(x)
        assert len(values) == 2, "x must be a 2-tuple"
        return (int(values[0]), int(values[1]))
    raise TypeError(f"Expected int or tuple[int, int], got {type(x)}")


class FlexiPatchEmbed(nn.Module):
    """Flexible patch embedding nn.Module.
    
    灵活的图像块嵌入模块，支持动态调整patch size。
    
    核心功能：
    1. 将2D图像（或多时间序列图像）转换为patch embeddings
    2. 支持通过插值动态调整patch size，实现多尺度特征提取
    3. 提供Linear和Conv2d两种投影方式，兼顾性能和兼容性
    
    工作原理：
    - 输入形状：[batch, height, width, (time), channels]
    - 输出形状：[batch, h_patches, w_patches, (time), embedding_dim]
    - 通过interpolation调整输入分辨率以匹配不同的patch size
    - 使用rearrange进行维度变换，适配Linear或Conv2d操作
    """

    def __init__(
        self,
        modality_spec: ModalitySpec,
        patch_size_at_16: int | tuple[int, int],
        in_chans: int = 3,
        embedding_size: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        interpolation: str = "bicubic",
        antialias: bool = True,
        use_linear_patch_embed: bool = True,
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes.

        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
        by https://github.com/bwconrad/flexivit/

        Args:
            modality_spec: The modality spec for this modality
                模态规格配置，包含image_tile_size_factor等参数
            patch_size_at_16: Base patch size. i.e the size of the parameter buffer at a resolution of 16
                基准patch size（在16倍分辨率下的尺寸），实际使用时会乘以modality_spec.image_tile_size_factor
            in_chans: Number of input image channels
                输入图像的通道数（如RGB为3，多光谱可能更多）
            embedding_size: Network embedding dimension size
                输出嵌入向量的维度
            norm_layer: Optional normalization layer
                可选的归一化层，如LayerNorm
            bias: Whether to use bias in convolution
                是否在卷积/线性投影中使用偏置项
            interpolation: Resize interpolation type
                插值方法，如'bicubic'、'bilinear'等，用于调整图像分辨率
            antialias: Whether to apply antialiasing resizing
                是否在resize时应用抗锯齿，避免混叠效应
            use_linear_patch_embed: If True, use nn.Linear (reshape + matmul via cuBLAS GEMM).
                If False, use nn.Conv2d (required to load checkpoints trained before this flag existed).
                选择投影方式：
                - True: 使用nn.Linear（reshape后矩阵乘法），利用cuBLAS GEMM加速，适合小通道数
                - False: 使用nn.Conv2d，用于加载旧版本训练的checkpoint
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.use_linear_patch_embed = use_linear_patch_embed

        self.modality_spec = modality_spec
        # 根据模态的tile size因子调整实际patch size
        # 例如：不同分辨率的卫星影像需要不同的patch size来保持合理的token数量
        self.patch_size = _to_2tuple(
            patch_size_at_16 * modality_spec.image_tile_size_factor
        )

        p_h, p_w = self.patch_size
        if use_linear_patch_embed:
            # Reshape patches to (p1 p2 c) then project — hits cuBLAS GEMM (always fast
            # on TensorCores) vs Conv2d which hits slow cuDNN paths for small in_chans.
            # 使用Linear投影：将每个patch展平为向量，然后通过全连接层映射到embedding空间
            # 优势：对于小通道数，cuBLAS GEMM比cuDNN的Conv2d更快
            self.proj = nn.Linear(in_chans * p_h * p_w, embedding_size, bias=bias)
            # Keep PyTorch's default nn.Linear initialization (kaiming_uniform_) for
            # patch projection to match prior Conv2d behavior; overriding this with
            # encoder-level Xavier init correlated with a PASTIS regression.
            # 保持PyTorch默认的kaiming_uniform_初始化，与之前Conv2d行为一致
            # 避免使用Xavier初始化（曾导致PASTIS回归任务性能下降）
            self.proj._skip_custom_init = True
        else:
            # 使用Conv2d投影：传统方式，用于兼容旧checkpoint
            self.proj = nn.Conv2d(
                in_chans,
                embedding_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=bias,
            )
        self.norm = norm_layer(embedding_size) if norm_layer else nn.Identity()
        self.interpolation = interpolation
        self.antialias = antialias

    def _resolve_patch_size(
        self, patch_size: int | tuple[int, int] | None
    ) -> tuple[int, int]:
        """Resolve the effective patch size, applying the modality tile size factor.
        
        解析有效的patch size，应用模态的tile size因子。
        
        处理逻辑：
        1. 如果patch_size为None，返回默认的自我.patch_size
        2. 否则，将输入的patch_size乘以modality_spec.image_tile_size_factor
        3. 确保返回值为二元元组格式
        
        Args:
            patch_size: 输入的patch size，可以是None、整数或二元元组
            
        Returns:
            解析后的patch size元组（已应用tile size因子）
        """
        if not patch_size:
            return self.patch_size
        if isinstance(patch_size, tuple):
            # 对元组的每个元素都应用tile size因子
            patch_size = (
                patch_size[0] * self.modality_spec.image_tile_size_factor,
                patch_size[1] * self.modality_spec.image_tile_size_factor,
            )
        else:
            # 对标量应用tile size因子
            patch_size = patch_size * self.modality_spec.image_tile_size_factor
        resolved = _to_2tuple(patch_size)
        assert isinstance(resolved, tuple) and len(resolved) == 2
        return resolved

    def _project_linear(
        self,
        x: Tensor,
        h_patches: int,
        w_patches: int,
        batch_size: int,
        has_time_dim: bool,
        num_timesteps: int,
    ) -> Tensor:
        """Project patches using nn.Linear (reshape → cuBLAS GEMM → reshape).
        
        使用nn.Linear进行patch投影，通过reshape触发cuBLAS GEMM优化。
        
        数据流：
        1. 输入：[batch*time, channels, H, W]
        2. Rearrange为：[batch*time, h_patches*w_patches, patch_h*patch_w*channels]
           - 将每个patch展平为一个向量
        3. Linear投影：[batch*time, h_patches*w_patches, embedding_size]
        4. 根据是否有时间维度，rearrange回合适的形状
        
        Args:
            x: 输入张量，形状为[batch*time, channels, H, W]
            h_patches: 高度方向的patch数量
            w_patches: 宽度方向的patch数量
            batch_size: 批次大小（不含时间维度）
            has_time_dim: 是否包含时间维度
            num_timesteps: 时间步数量
            
        Returns:
            投影后的张量：
            - 有时间维度：[batch, h_patches, w_patches, time, embedding_size]
            - 无时间维度：[batch, h_patches, w_patches, embedding_size]
        """
        p_h, p_w = self.patch_size
        # 将空间维度和通道维度重组：提取每个patch并展平
        # b c (h p1) (w p2) -> b (h w) (p1 p2 c)
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p_h, p2=p_w)
        # 通过Linear层投影到embedding空间
        x = self.proj(x)
        if has_time_dim:
            # 恢复时间维度：(b t) (h w) d -> b h w t d
            return rearrange(
                x,
                "(b t) (h w) d -> b h w t d",
                b=batch_size,
                t=num_timesteps,
                h=h_patches,
                w=w_patches,
            )
        # 无时间维度：b (h w) d -> b h w d
        return rearrange(x, "b (h w) d -> b h w d", h=h_patches, w=w_patches)

    def _project_conv(
        self,
        x: Tensor,
        batch_size: int,
        has_time_dim: bool,
        num_timesteps: int,
    ) -> Tensor:
        """Project patches using nn.Conv2d (for loading pre-linear checkpoints).
        
        使用nn.Conv2d进行patch投影，用于兼容旧版本训练的checkpoint。
        
        数据流：
        1. 输入：[batch*time, channels, H, W]
        2. Conv2d投影：[batch*time, embedding_size, H_out, W_out]
           - kernel_size和stride都等于patch_size，实现非重叠分块
        3. 根据是否有时间维度，rearrange回合适的形状
        
        Args:
            x: 输入张量，形状为[batch*time, channels, H, W]
            batch_size: 批次大小（不含时间维度）
            has_time_dim: 是否包含时间维度
            num_timesteps: 时间步数量
            
        Returns:
            投影后的张量：
            - 有时间维度：[batch, h_patches, w_patches, time, embedding_size]
            - 无时间维度：[batch, h_patches, w_patches, embedding_size]
        """
        # Conv2d自动完成分块和投影：b c h w -> b d h_out w_out
        x = self.proj(x)
        if has_time_dim:
            _, d, h, w = x.shape
            # 恢复时间维度：(b t) d h w -> b h w t d
            return rearrange(
                x,
                "(b t) d h w -> b h w t d",
                b=batch_size,
                t=num_timesteps,
                h=h,
                w=w,
            )
        # 无时间维度：b d h w -> b h w d
        return rearrange(x, "b d h w -> b h w d")

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor:
        """Forward pass for the FlexiPatchEmbed module.

        Args:
            x: Input tensor with shape [b, h, w, (t), c]
                输入张量，形状为[batch, height, width, (time), channels]
                - 有时间维度：5维张量
                - 无时间维度：4维张量
            patch_size: Patch size to use for the embedding. If None, uses the base patch size.
                使用的patch size，如果为None则使用默认的self.patch_size
                支持动态调整，实现多尺度特征提取
                
        Returns:
            嵌入后的张量，形状为[batch, h_patches, w_patches, (time), embedding_size]
            
        处理流程：
        1. 检测输入是否包含时间维度
        2. 将数据重组为[b*t, c, h, w]格式（合并时间和批次维度）
        3. 解析目标patch size
        4. 如果patch size与默认值不同，通过interpolation调整输入分辨率
        5. 计算输出的patch网格尺寸
        6. 根据use_linear_patch_embed选择投影方式（Linear或Conv2d）
        7. 应用归一化层
        8. 恢复原始维度结构
        """
        batch_size = x.shape[0]
        # 检测是否包含时间维度：5维表示有时间轴
        has_time_dim = len(x.shape) == 5
        num_timesteps = x.shape[3] if has_time_dim else 0

        # 将数据重组为Conv2d/Linear友好的格式
        if has_time_dim:
            # b h w t c -> (b t) c h w：合并批次和时间维度，通道移到第二维
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            # b h w c -> b c h w：仅移动通道维度
            x = rearrange(x, "b h w c -> b c h w")

        # 解析实际使用的patch size（应用tile size因子）
        patch_size = self._resolve_patch_size(patch_size)

        # 如果目标patch size与默认值不同，需要调整输入分辨率
        if patch_size != self.patch_size:
            shape = x.shape[-2:]  # 当前空间尺寸 [H, W]
            # 计算新的空间尺寸，确保能被patch size整除
            new_shape = (
                shape[0] // patch_size[0] * self.patch_size[0],
                shape[1] // patch_size[1] * self.patch_size[1],
            )
            # 通过插值调整分辨率
            # 关键：这里调整的是输入图像的分辨率，而非patch size本身
            # 目的是让不同patch size下产生的patch数量保持一致或合理
            x = F.interpolate(
                x, size=new_shape, mode=self.interpolation, antialias=self.antialias
            )

        # 计算输出的patch网格尺寸
        p_h, p_w = self.patch_size
        h_patches, w_patches = x.shape[2] // p_h, x.shape[3] // p_w

        # 选择投影方式
        if self.use_linear_patch_embed:
            # 使用Linear投影（推荐，性能更好）
            x = self._project_linear(
                x, h_patches, w_patches, batch_size, has_time_dim, num_timesteps
            )
        else:
            # 使用Conv2d投影（兼容旧checkpoint）
            x = self._project_conv(x, batch_size, has_time_dim, num_timesteps)

        # 应用归一化层
        return self.norm(x)


class FlexiPatchReconstruction(nn.Module):
    """Flexible patch reconstruction nn.Module.
    
    灵活的图像块重建模块，是FlexiPatchEmbed的逆操作。
    
    核心功能：
    1. 将patch embeddings重建为2D图像（或多时间序列图像）
    2. 支持动态调整patch size，与FlexiPatchEmbed配合实现可变分辨率重建
    3. 使用ConvTranspose2d进行上采样和反投影
    
    应用场景：
    - 自编码器的解码器部分
    - Masked Autoencoder的重建头
    - 生成模型中的上采样模块
    
    工作原理：
    - 输入形状：[batch, h_patches, w_patches, (time), embedding_dim]
    - 输出形状：[batch, height, width, (time), channels]
    - 通过ConvTranspose2d将embedding映射回像素空间
    - 支持通过插值调整输出分辨率以匹配不同的patch size
    """

    def __init__(
        self,
        max_patch_size: int | tuple[int, int],
        out_chans: int = 3,
        embedding_size: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """Patch embeding to 2d image reconstruction w/ flexible patch sizes.

        Args:
            max_patch_size: Base patch size. i.e the size of the parameter buffer
                最大patch size，即参数缓冲区的尺寸
                这是ConvTranspose2d的kernel_size和stride
            out_chans: Number of out image channels
                输出图像的通道数（如RGB为3）
            embedding_size: Network embedding dimension size
                输入嵌入向量的维度
            norm_layer: Optional normalization layer
                可选的归一化层
            bias: Whether to use bias in convolution
                是否在转置卷积中使用偏置项
            interpolation: Resize interpolation type
                插值方法，用于调整patch内部分辨率
            antialias: Whether to apply antialiasing resizing
                是否在resize时应用抗锯齿
        """
        super().__init__()

        self.embedding_size = embedding_size

        self.max_patch_size = _to_2tuple(max_patch_size)

        # 使用转置卷积进行上采样和反投影
        # kernel_size和stride都等于max_patch_size，实现从patch到图像的扩展
        self.proj = nn.ConvTranspose2d(
            embedding_size,
            out_chans,
            kernel_size=max_patch_size,
            stride=max_patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embedding_size) if norm_layer else nn.Identity()
        self.interpolation = interpolation
        self.antialias = antialias

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        """Resize the input tensor to the target shape.

        Args:
            x: Input tensor
                输入张量
            shape: Target shape
                目标形状 (height, width)

        Returns:
            Resized tensor
                调整后的张量
        """
        # 添加batch和channel维度以满足F.interpolate的输入要求
        # x原本是2D的，需要先变成[1, 1, H, W]格式
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        # 移除添加的维度，恢复原始维度结构
        return x_resized[0, 0, ...]

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        """Forward pass for the FlexiPatchReconstruction module.

        Args:
            x: Input tensor with shape [b, h, w, (t), d]
                输入张量，形状为[batch, h_patches, w_patches, (time), embedding_dim]
                - 有时间维度：5维张量
                - 无时间维度：4维张量
            patch_size: Patch size to use for the reconstruction. If None, the base patch size
                will be used.
                重建时使用的patch size，如果为None则使用max_patch_size
                允许动态调整输出分辨率
                
        Returns:
            重建后的张量，形状为[batch, height, width, (time), channels]
            
        处理流程：
        1. 检测输入是否包含时间维度
        2. 将数据重组为[(b*t), embedding_dim, h_patches, w_patches]格式
        3. 通过ConvTranspose2d上采样：从[h_patches, w_patches]扩展到更大的空间尺寸
        4. 如果patch_size与max_patch_size不同，对每个patch内部进行插值调整
        5. 恢复时间维度（如果存在）
        6. 应用归一化层
        7. 返回重建的图像
        
        关键技术点：
        - 当patch_size < max_patch_size时，需要对每个patch内部进行降采样
        - 通过rearrange将patch维度分离出来，单独对每个patch插值
        - 最后重新组合成完整的图像
        """
        # x has input shape [b, h, w, (t), d]
        # 检测时间维度并提取形状信息
        if len(x.shape) == 4:
            has_time_dimension = False
            b, h, w, d = x.shape
            t = 1
        else:
            has_time_dimension = True
            b, h, w, t, d = x.shape

        if not patch_size:
            # During evaluation use base patch size if not specified
            # 评估时如果没有指定patch_size，使用最大patch size
            patch_size = self.max_patch_size

        patch_size = _to_2tuple(patch_size)

        # 将数据重组为ConvTranspose2d友好的格式
        if has_time_dimension:
            # b h w t d -> (b t) d h w：合并批次和时间维度
            x = rearrange(x, "b h w t d -> (b t) d h w", b=b, t=t)
        else:
            # b h w d -> b d h w：仅移动通道维度
            x = rearrange(x, "b h w d -> b d h w")

        # 通过转置卷积进行上采样和反投影
        # 输入：[(b*t), embedding_dim, h_patches, w_patches]
        # 输出：[(b*t), out_chans, h_patches*max_patch_h, w_patches*max_patch_w]
        x = self.proj(x)

        # 如果目标patch size与最大patch size不同，需要调整每个patch内部的分辨率
        if patch_size != self.max_patch_size:
            # 将输出分解为patch级别：分离出patch内部的空间维度
            # b c (h p_h) (w p_w) -> b h w c p_h p_w
            # 其中p_h和p_w是max_patch_size，表示每个patch的内部尺寸
            x = rearrange(
                x,
                "b c (h p_h) (w p_w) -> b h w c p_h p_w",
                p_h=self.max_patch_size[0],
                p_w=self.max_patch_size[1],
            )
            bl, hl, wl, cl = x.shape[:4]  # 保存batch、h、w、channel维度
            # 将所有patch展平，以便批量处理
            x = rearrange(x, "b h w c p_h p_w -> (b h w) c p_h p_w")
            # 对每个patch内部进行插值，调整到目标patch_size
            # 例如：从16x16降到8x8，或从8x8升到16x16
            x = F.interpolate(
                x, patch_size, mode=self.interpolation, antialias=self.antialias
            )
            # 重新组合成完整图像
            # (b h w) c p_h p_w -> b c (h p_h) (w p_w)
            x = rearrange(
                x, "(b h w) c p_h p_w -> b c (h p_h) (w p_w)", b=bl, h=hl, w=wl
            )

        # 恢复时间维度（如果存在）
        if has_time_dimension:
            # (b t) c h w -> b h w t c
            x = rearrange(x, "(b t) c h w -> b h w t c", b=b, t=t)
        else:
            # b c h w -> b h w c
            x = rearrange(x, "b c h w -> b h w c")

        # 应用归一化层
        x = self.norm(x)

        return x
