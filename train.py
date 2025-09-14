import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.independentVL
import trainers.promptsrc
import trainers.promptkd
import trainers.sple_promptkd  # PromptKD+DPC
import trainers.promptkd_infer  # Real-time inference head for PromptKD+DPC

# import trainers.coop_stats
# import trainers.promptsrc_stats


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = ""

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for PromptSRC
    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1


    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting(J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TEST.NO_TEST = False

    # KD
    # cfg.MODEL.BACKBONE.TEACHER_NAME = "ViT/L-14"
    # cfg.MODEL.BACKBONE.PROJECT_LAYER = 2
    # cfg.MODEL.BACKBONE.CE_WEIGHT = 0.0

    cfg.TRAINER.MODAL = "base2novel"
    cfg.TRAINER.PROMPTKD = CN()
    cfg.TRAINER.PROMPTKD.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTKD.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTKD.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTKD.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTKD.PROJECT_LAYER = 2
    cfg.TRAINER.PROMPTKD.CE_WEIGHT = 0.0
    cfg.TRAINER.PROMPTKD.KD_WEIGHT= 1.0
    cfg.TRAINER.PROMPTKD.TEMPERATURE = 1.0
    cfg.TRAINER.PROMPTKD.TEACHER_NAME = "ViT-L/14"

    ''' === [DPC] CONFIGS === '''
    cfg.SPLE = CN()
    # Fine-tuning ckeckpoint and epoch of pre-tuned backbone (e.g., CoOp). Must be provided before tuning on DPC.
    cfg.SPLE.BACK_CKPT_PATH = "../output/caltech101-N3-Final/CoOp/vit_b16_ep100_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-100"
    cfg.SPLE.BACK_CKPT_EPOCH = 100

    cfg.SPLE.INFER_TOPK = 8  # Top-K for dynamic hard negative sampler
    cfg.SPLE.INFER_NONREPEAT = True  # No repeat object in a mini-batch
    cfg.SPLE.PIC_LIB = "../DATA/SPLE_database/SPLE_Caltech101.json"
    cfg.SPLE.LOSS_KD_WEIGHT = 0.5  # [DPC_PromptKD] Weight for fusing DPC_loss (L_cl) and PromptKD_loss (L_kd)
    cfg.SPLE.KD_INFER = ""  # [DPC_PromptKD] Name of inference head

    cfg.SPLE.SPLE_TRAINER = CN()
    cfg.SPLE.SPLE_TRAINER.SPLE_INIT = True  # [DPC] Use DPC to init
    cfg.SPLE.SPLE_TRAINER.SPLE_BASE_TRAIN = True  # [DPC] Use same base classes (keep 'True' to avoid data leakage)

    cfg.SPLE.STACK = CN()
    cfg.SPLE.STACK.MODE = "void"  # [DISCARD] DO NOT CHANGE
    cfg.SPLE.STACK.WEIGHT = 0.5  # [DPC] Weight for base
    cfg.SPLE.STACK.WEIGHT_FOR_NEW = 0.2  # [DPC] Weight for new
    cfg.SPLE.STACK.LOOP_DEPTH = 0  # [DISCARD] DO NOT CHANGE

    # [MAO] Configs of MAO: Efficient Model-Agnostic Optimization of Prompt Tuning for Vision-Language Models
    # Link of paper: https://arxiv.org/abs/2503.18160
    cfg.MAO = CN()
    cfg.MAO.PIC_LIB = "DATA/SPLE_database/SPLE_OxfordFlowers.json"
    cfg.MAO.INFER_TOPK = 8
    cfg.MAO.BASE_CKPT = "output/MAO_CoOp/base2new/base_task/caltech101/vit_b16_bs4_ep10_lr35_S8/seed1/prompt_learner/model.pth.tar-10"
    cfg.MAO.BACK_CKPT_EPOCH = 10
    
    
def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
