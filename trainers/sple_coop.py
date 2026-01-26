import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .base import TrainerX
from .optim import build_optimizer

_tokenizer = _Tokenizer()

# [StackSPLE] import
from dassl.data.transforms.transforms import build_transform
from dassl.engine import build_trainer
import json
import random
from dassl.utils import read_image
import trainers.coop_stats


# [SPLE-MaPLe 重构] 用于 negative sampling 阶段, 转换从图片 database 采样得到的图片数据
def transform_image(cfg, img0, transform):
    def _transform_image(tfm, img0):
        img_list = []
        for k in range(1):
            img_list.append(tfm(img0))
        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

    output = {}
    # 引入 tfm 进行图片读取操作
    if isinstance(transform, (list, tuple)):
        for i, tfm in enumerate(transform):
            img = _transform_image(tfm, img0)
            keyname = "img"
            if (i + 1) > 1:
                keyname += str(i + 1)
            output[keyname] = img
    else:
        img = _transform_image(transform, img0)
        output["img"] = img  # [3, 224, 224]

    return output


# [SPLE] 将 DataLoader 获取的图像绝对路径，切分为仅包含相对路径的后缀 (suffix), 与剩余部分的前缀 (prefix).
# 需要同时考虑 win 系统 (D:\\XXX\\dolphin/image_0011.jpg) 与 Linux 系统 (/usr/XXX/dolphin/image_0011.jpg)
# 参考 database 标注文件的 / 数量
def split_img_abs_path(abs_path, ref_path):
    split_sum = ref_path.count("/")  # 以 database 中的路径名作为参考，统计 "/" 的数量
    if "\\" in abs_path:
        split_result = abs_path.rsplit("\\", 1)  # 依据最后一个 "\\" 进行切分
        path_prefix = split_result[0]
        path_suffix = split_result[1]
    elif "r'\'" in abs_path:
        split_result = abs_path.rsplit("r'\'", 1)  # 依据最后一个 "\" 进行切分
        path_prefix = split_result[0]
        path_suffix = split_result[1]
    else:
        split_result = abs_path.rsplit("/", split_sum + 1)  # 依据倒数第 n+1 个 "/" 进行切分
        path_prefix = split_result[0]
        path_suffix = split_result[1]
        if len(split_result) > 1:
            for split_id in range(2, len(split_result)):
                path_suffix = path_suffix + "/" + split_result[split_id]

    return path_prefix, path_suffix


# [SPLE_CoOp 重构] 读取 CoOp 的原始 prompt vector, 作为 stack prompt 初始化工具
# 输出 2 个参数: 维度为 (n_ctx, dim) 的 ctx prompt 向量, 和预微调的完整 prompt learner
def load_backbone_prompt_vector(cfg):
    upper_path = cfg.SPLE.BACK_CKPT_PATH  # 配置文件，引入 CoOp 训练好的 model.pth.tar-50 权重文件
    ckpt_epoch = cfg.SPLE.BACK_CKPT_EPOCH
    model_path = upper_path + "/prompt_learner/model.pth.tar-" + str(ckpt_epoch)  # CoOp 专属路径

    prompt_learner = torch.load(model_path, map_location="cuda")  # 读取权重
    # 从检查点中读取 prefix 与 suffix 信息，组成完整 prompt
    ctx = prompt_learner["state_dict"]["ctx"]

    return ctx, prompt_learner


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype

        # [StackSPLEForNew] 如果使用 base/new 方式训练，则强制使 SPLE.SPLE_TRAINER.SPLE_BASE_TRAIN 为 False
        self.base2new = cfg.DATASET.SUBSAMPLE_CLASSES
        self.sple_stack_weight_for_new = cfg.SPLE.STACK.WEIGHT_FOR_NEW
        sple_init = cfg.SPLE.SPLE_TRAINER.SPLE_INIT  # 是否使用 SPLE 方式进行初始化
        self.sple_stack_mode = cfg.SPLE.STACK.MODE  # [StackSPLE] void 为不堆叠, simple 为仅堆叠 1 次, loop 为堆叠 N 次
        self.sple_stack_weight = cfg.SPLE.STACK.WEIGHT  # [StackSPLE] stack prompt 占据的权重
        self.sple_stack_depth = cfg.SPLE.STACK.LOOP_DEPTH  # [StackSPLE] loop 模式下，SPLE 模块堆叠数量
        self.cfg = cfg

        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # [StackSPLE_PromptSRC] SPLE 初始化方法
        # [StackSPLE_MaPLe] 如果使用 converse 或 simple 方式，则进入 stack 模式
        # 在构建可学习参数时，为 stack prompt 添加权重并与预微调 prompt 结合，形成 mixed_ctx prompt，会保存到权重中；
        # 但在向 prompt learner 传递的时候，converse 模式会通过还原，仅传递 stack prompt，推理过程时才使用 mixed_ctx prompt.
        # simple 模式将不还原，即直接使用 mixed_prompt (实际上也就是 CoOp 预微调权重) 进行 forward_backward 与推理。
        if "converse" in self.sple_stack_mode or "simple" in self.sple_stack_mode:
            # [StackSPLE_CoOp] 读入预微调的 ctx
            self.pretuned_ctx, _ = load_backbone_prompt_vector(cfg)

            # 终端中打印说明信息
            if "converse" in self.sple_stack_mode:
                print("[StackSPLE] Construct ctx prompt in --converse-- mode: use stack prompt for prompt tuning, ",
                      "and save & use mixed prompt for inference.")
            else:
                print("[StackSPLE] Construct ctx prompt by mixed-ctx using --simple-- stack settings")

            print("[StackSPLE] Stack method:", self.sple_stack_mode, "; Stack weight:", self.sple_stack_weight)
            print(
                "[StackSPLE-CoOp] Params to stack: prompt_learner.ctx")

            if sple_init:
                # [StackSPLE_CoOp] sple_init 中，由于使用预微调权重进行初始化，因此加权后的 prompt 也依旧是预微调权重
                mixed_ctx = self.pretuned_ctx
            else:
                # [SPLE_PromptSRC] 如果并未使用预微调权重进行初始化，则将以 PromptSRC 方式随机初始化的参数，与预微调权重加权
                mixed_ctx = (self.sple_stack_weight * ctx_vectors + (
                        1 - self.sple_stack_weight) * self.pretuned_ctx).type(dtype)

            # [SPLE_PromptSRC] 使用 nn.Parameter() 包裹并回传所有加权 prompt vector
            self.ctx = nn.Parameter(mixed_ctx)

        # [MaPLe 重构] 不使用加权方法，直接调用原始 MaPLe 方式
        else:
            print("[StackSPLE-CoOp] No stack mode applied, following original CoOp to init.")
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        # self.register_buffer("tokenized_prompts", tokenized_prompts)  # [SPLE] 把 tokenized_prompts 也输出到权重文件里

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        # [StackSPLE] 若在 converse 模式下运行，则将 ctx 还原为 stack prompt, 输入网络进行训练
        # 在 converse 模式下, 向 pth 输出的 / val 过程中使用的, 依旧是 mixed_prompt
        if "converse" in self.sple_stack_mode:
            ctx = (self.ctx - (1 - self.sple_stack_weight) * self.pretuned_ctx) * (1 / self.sple_stack_weight)
        # [PromptSRC 重构] 原始 CoOp 的 ctx
        else:
            ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class CustomSPTCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, sple_label=None):
        image_features = self.image_encoder(image.type(self.dtype))  # torch.Size(bs*TopK, 512)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)  # torch.Size(n_cls, 512)

        # SPLE 训练阶段, 对于 parse_sple_batch_train() 获取的 bs*TopK 个图文对, 获取对于图像与对于文本的两个 logits
        if self.prompt_learner.training:
            # 在全部类别组成的 text_features 中, 依据 sple_label 的 id, 单独抽取出与 mini-batch 中
            # 每个样本的 hard negative id 对应索引的张量，构成新的 text features, 尺寸为 [bs*TopK, 512]
            text_features = text_features[sple_label.tolist()]
            # 训练阶段读入的 image feature, 尺寸亦为 [bs*TopK, 512]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_per_img = logit_scale * image_features @ text_features.t()  # torch.Size(bs*TopK, TopK*bs)
            logits_per_text = logits_per_img.t()  # torch.Size(TopK*bs, bs*TopK)

            return logits_per_img, logits_per_text

        else:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return logits


@TRAINER_REGISTRY.register()
class StackSPLE_CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transform_img = build_transform(cfg, is_train=True)  # [SPLE] 引入额外图像转换编码方法

        # [StackSPLEForNew] 如果使用 simple 堆叠模式 (pth 文件里保存的是 mixed_prompt), 则需要将其还原为 stack prompt
        self.sple_stack_weight = cfg.SPLE.STACK.WEIGHT  # stack prompt 叠加权重
        self.sple_stack_weight_for_new = cfg.SPLE.STACK.WEIGHT_FOR_NEW  # base2new 推理时 stack prompt 所占权重比例

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # [StackSPLEForNew] 如果使用 base/new 方式训练，则强制使 SPLE.SPLE_TRAINER.SPLE_BASE_TRAIN 为 False
        self.base2new = cfg.DATASET.SUBSAMPLE_CLASSES
        print("self.base2new", self.base2new)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP in ***SPLE*** setting")
        self.model = CustomSPTCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        # self.model.to(self.device)
        # # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.model.to(self.device)
        self.optim, infos = build_optimizer(self.model, cfg.OPTIM)

        if infos is not None:
            print('Learning rate of parameters:')
            for info in infos:
                print('lr: {}, layers: {}'.format(info['lr'], info['layers']))

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        # [MaPLe 重构] 初始化 MaPLe 的 inference backbone
        self.inference_trainer = self.build_inference_backbone(cfg)

    def forward_backward(self, batch):
        # [SPLE] 引入 parse_sple_batch_train(): 对于长度为 [bs] 的 mini-batch, 使用 SPLE 方法获取 hard negative 图文对, 使长度变为 [bs*TopK]
        image, label = self.parse_sple_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # [SPLE] 对来自图像与文本的 logits 分别做交叉熵，并平均损失
            logits_per_img, logits_per_text = self.model(image, label)
            # print("logits_per_img", logits_per_img, "logits_per_text", logits_per_text)
            # 对 mini-batch 做标签, [0,1,2,...,bs*TopK]
            label_ids = torch.arange(label.size(0), device=self.device).long()
            loss = (F.cross_entropy(logits_per_img, label_ids) +
                    F.cross_entropy(logits_per_text, label_ids)
                    ) / 2
            # 反向传播 loss
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item()}  # 不计算 acc

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    # [SPLE] 引入 PromptSRC inference 模型 ("PromptSRC"), 用于样本筛选
    @torch.no_grad()
    def build_inference_backbone(self, cfg):
        print("==== [SPLE-CoOp] build a seprate CoOp backbone model for inference ====")
        trainer = build_trainer(cfg, name="CoOpStats")
        model_dir = cfg.SPLE.BACK_CKPT_PATH  # [V4]
        trainer.load_model(model_dir, epoch=cfg.SPLE.BACK_CKPT_EPOCH)  # [V4]
        trainer.set_model_mode("eval")

        # 测试是否成功调用
        clip_model = trainers.coop.load_clip_to_cpu(cfg)
        logit_scale = clip_model.logit_scale
        print("=====logit_scale=====", logit_scale)

        return trainer

    # [SPLE] 对于给定的 batch, 对逐个 img 进行 prompt learner 推理, 并根据推理得到的 hard negative object 随机抽取图片
    # 最终形成包含 ground-truth 与 Top-K hard negative object 的 mini-batch
    # 为实现交叉熵，nonrepeat 模式下，若 Top-K 推理得到的 object 已经被包含在整个 mini-batch 中先前得到的 objects 内，则排除
    # ([non-repeat] 设定) 保证整个 mini-batch 输出的 object 列表中，不存在重复项
    def parse_sple_batch_train(self, batch):
        input = batch["img"]  # torch.Size([bs,3,224,224])
        label = batch["label"]  # torch.Size([bs])
        img_path = batch["impath"]  # torch.Size([bs])
        # 读取并扩展 cfg 配置
        cfg = self.cfg
        class_label = self.dm.dataset.classnames  # 列表格式
        topk_sum = cfg.SPLE.INFER_TOPK  # 采样 hard negative object 的数量
        pic_lib = cfg.SPLE.PIC_LIB  # 图片 database 路径

        with torch.no_grad():
            inference_trainer = self.inference_trainer  # [MaPLe重构] 初始化 CoOp 的 inference backbone
            # 初始化空张量
            input_sple = torch.empty(0, 3, 224, 224)
            label_sple = torch.empty(0)
            # 存储该 Mini-batch 出现过的所有 object
            # 使用 mini-batch 的全部正样本 id 初始化。但注意，因为 id 排序问题，不可直接使用此列表作为 label_sple.
            objects_in_batch = label.tolist()

            # [MaPLe重构] 对于整个 batch 的图像，使用 CoOp backbone 推理
            # text_feats: [n_cls, 512]; img_feats: [bs, 512]; batch_similarity: [bs, n_cls]
            text_feats, img_feats = inference_trainer.model_inference(input.type(self.model.dtype).to(self.device))
            batch_similarity = (100.0 * img_feats @ text_feats.T).softmax(dim=-1)

            # 对于 batch 中的每个 input 图片与 label, 进行 Top-K 推理
            for sample_id in range(0, input.size(0)):
                # [MaPLe重构] 提取 batch 中第 sample_id 个相似度, indices 为对应 id, values 为对应相似度数值。
                # 两者尺寸均为 [n_topk]
                values, indices = batch_similarity[sample_id].topk(topk_sum)

                # 实体筛选: 采样除了 gt 之外的 Top K-1 个负样本, 作为 hard-negative objects (列表中不包含 gt)
                hn_labels_before_selection = []
                hn_labels = []
                for value, index in zip(values, indices):
                    if index != label[sample_id] and len(hn_labels) < topk_sum - 1:
                        hn_labels_before_selection.append(index)

                # 继续筛选: 将结果与该 Mini-batch 已经出现过的所有 object 对比
                # 如果新的 object 在先前从未出现过 (包括正样本与负样本)，才将其添加到 mini-batch 中
                for item in hn_labels_before_selection:
                    if item not in objects_in_batch:
                        hn_labels.append(item)
                        objects_in_batch.append(item)

                # 如果 hn_labels 通过上述方法没有查询到任何值，则从 objects_in_batch 外随机抽取 2 个 objects
                if len(hn_labels) < 2:
                    for step in range(0, len(class_label) - 2):
                        neg_label = random.randint(1, len(class_label) - 2)
                        if neg_label not in objects_in_batch and len(hn_labels) < 2:
                            hn_labels.append(neg_label)
                            objects_in_batch.append(neg_label)
                        elif len(hn_labels) < 2:
                            continue
                        else:
                            break

                # print("===original hn_labels===", hn_labels_before_selection)
                # print("===selected hn_labels===", hn_labels)

                # 图片采样: 使用 hn_labels 作为 query, 在图片资源库中，从对应 label 的图像列表中，随机采样正图片
                hn_pic_paths = []
                with open(pic_lib) as f:
                    pics_for_selection = json.load(f)
                    '''
                    pics_for_selection 对应的字典形如:
                    {
                        'train': [{'face': [0, ['1.jpg', '2.jpg']], 'leopard': [1, ['3.jpg', '4.jpg']], ... }],
                        'val': [{'face': [0, ['5.jpg', '6.jpg']], 'leopard': [1, ['7.jpg', '8.jpg']], ... }],
                        'train_obj_list': ['face', 'leopard', ...],
                        'val_obj_list': ['face', 'leopard', ...]
                    }
                    其中，'train' 与 'val' 键值的列表长度恒为 1.
                    '''
                    for obj_id in hn_labels:
                        hn_obj_name = class_label[obj_id]  # 获取类名
                        pic_list = pics_for_selection["train"][0].get(hn_obj_name)  # 通过搜索类名, 获取图片列表
                        random_pic_path = random.choice(pic_list[1])  # 在图片列表中，随机选取一张图片
                        hn_pic_paths.append(random_pic_path)

                # 图片转换: 读取图片, 并转换为 尺寸为 [3,224,224] 的 CLIP 标准输入形式
                input_for_concat = input[sample_id].unsqueeze(0).to(self.device)  # 建立用于与 hn 拼接的张量, [1,3,224,224]
                # 以 database 的图片相对路径 (random_pic_path) 作为参考, 提取绝对路径的前半部分 img_path_prefix
                # [MaPLe 重构] EuroSAT 与 ImageNet 的数据结构，与常规不同，需要单独判断
                dataset_name = cfg.DATASET.NAME
                if dataset_name == "EuroSAT":
                    img_path_prefix, _ = split_img_abs_path(img_path[sample_id], "Highway/Highway_2417.jpg")
                elif dataset_name == "ImageNet":
                    img_path_prefix, _ = split_img_abs_path(img_path[sample_id], random_pic_path)
                else:
                    # 如绝对路径为 D:\\XXX\\dolphin/image_0011.jpg, 参考路径为 a/01.jpg, 则 img_path_prefix 为 'D:\\XXX'
                    img_path_prefix, _ = split_img_abs_path(img_path[sample_id], random_pic_path)

                for processing_img in hn_pic_paths:
                    img0 = read_image(img_path_prefix + "/" + processing_img)  # 使用完整路径读取图片
                    transformed_img = transform_image(cfg, img0, self.transform_img)["img"].to(
                        self.device)  # [MaPLe 重构] 转换图片
                    # 对转换完成的图片, 进行张量拼接: 在 input_for_concat 上逐个拼接 hn 图片, 最终尺寸为 [TopK, 3, 244, 244]
                    input_for_concat = torch.cat([input_for_concat,
                                                  transformed_img.unsqueeze(0)
                                                  ],
                                                 dim=0)

                # 将 hn_labels 与转化为张量的原正样本拼接, 尺寸为 [TopK]
                label_for_concat = torch.cat([label[sample_id].unsqueeze(0), torch.Tensor(hn_labels)], dim=0)
                # 将单个 sample 获取到的完整数据进行拼接
                # label 数据: 将尺寸为 [TopK] 的张量拼接到原张量上，最终尺寸为 [bs * TopK]
                label_sple = torch.cat([label_sple, label_for_concat], dim=0)
                # image 数据: 将尺寸为 [TopK, 3, 244, 244] 的张量拼接到原张量上，最终尺寸为 [bs * TopK, 3, 244, 244]
                input_sple = torch.cat([input_sple.to(self.device), input_for_concat], dim=0)

            # 将最终完成的数据，传入 device, 作为一个 mini-batch
            input_sple = input_sple.to(self.device)
            label_sple = label_sple.type(label.dtype).to(self.device)  # type(label.dtype) 转化为与原始输出相同的 type

        # print("==> label_sple", label_sple)
        # print("==> input_sple.size", input_sple.size(), "==> label_sple.size", label_sple.size())
        return input_sple, label_sple


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            if epoch < 0:
                all_model_files = os.listdir(osp.join(directory, name))
                all_model_files = [file_ for file_ in all_model_files if file_ != 'checkpoint']
                model_epochs = [int(file_.split('-')[-1]) for file_ in all_model_files]
                last_epoch = max(model_epochs)
                model_file = 'model.pth.tar-' + str(last_epoch)

            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "tokenized_prompts" in state_dict:
                del state_dict["tokenized_prompts"]

            # [V2-BUGFIX] 提取 ctx (stack prompt)
            stored_stack_prompt = state_dict["ctx"]
            # [StackSPLEForNew] new 时, 如果使用 simple 堆叠模式 (pth 文件里保存的是 mixed_prompt),
            # 则需要将其还原为 stack prompt, 随后使用 sple_stack_weight_for_new 权重，进行重新加权
            cfg = self.cfg
            sple_stack_mode = cfg.SPLE.STACK.MODE
            p_ctx, p_dict = load_backbone_prompt_vector(cfg)
            # [V2-BUGFIX] 增加 all 下新类的判断
            if self.base2new == "new" or (self.base2new == "all" and self.cfg.DATASET.NAME != "ImageNet"):
                state_dict = p_dict["state_dict"]
                # [SPLE] 删除权重文件中所有不可学习参数
                if "token_prefix" in state_dict:
                    del state_dict["token_prefix"]
                if "token_suffix" in state_dict:
                    del state_dict["token_suffix"]
                if "tokenized_prompts" in state_dict:
                    del state_dict["tokenized_prompts"]

                # 获取可学习参数的命名列表
                key_list = []
                for key in state_dict:
                    key_list.append(key)
                print(key_list)

                # [SPLE_PromptSRC] 将 simple 模式输出到权重的 mixed prompt 还原为 stack prompt
                if "simple" in sple_stack_mode:
                    print("[StackSPLEForNew_MaPLe] Convert mixed prompts in 'simple' mode to tuned stack prompts.")
                    mixed_ctx = stored_stack_prompt
                    stack_ctx = (mixed_ctx - (1 - self.sple_stack_weight) * p_ctx) * (1 / self.sple_stack_weight)
                    # converse 堆叠模式, 保存的就是 stack prompt 的 ctx
                else:
                    stack_ctx = stored_stack_prompt

                print("[StackSPLEForNew] Give weight for tuned unmixed stack prompt to inference on new class.")
                # 在 new 上推理，按照 sple_stack_weight_for_new 的权重，重新对 ctx 加权
                mixed_ctx_for_n = self.sple_stack_weight_for_new * stack_ctx + (
                            1 - self.sple_stack_weight_for_new) * p_ctx
                state_dict["ctx"] = mixed_ctx_for_n

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
