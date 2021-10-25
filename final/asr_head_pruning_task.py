import torch
import argparse
from espnet2.tasks.asr import ASRTask
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.bin.asr_inference import Speech2Text
from espnet2.train.distributed_utils import DistributedOption
from espnet2.torch_utils.device_funcs import to_device
from grad_tracer import TransformerTracer
from tqdm import tqdm


class ASRHeadPruningTask(ASRTask):
    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        super().add_task_arguments(parser)

        required = parser.get_default("required")
        required += ["pretrained_model_name"]

        group = parser.add_argument_group(description="Head-pruning related")
        group.add_argument('--pretrained_model_name', type=str)
        group.add_argument('--path_head_grad', type=str)

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRModel:
        model = super().build_model
        return model

    @classmethod
    def main(cls, args=None, cmd=None):
        if args is None:
            parser = cls.get_parser()
            args = parser.parse_args(cmd)

        set_all_random_seed(args.seed)
        # model = cls.build_model(args=args)
        # model = model.to(
        #     dtype=getattr(torch, args.train_dtype),
        #     device="cuda" if args.ngpu > 0 else "cpu",
        # )

        speech2text = Speech2Text.from_pretrained(
            args.pretrained_model_name,
            ctc_weight=0,
            device='cuda'
        )
        model = speech2text.asr_model
        model.train()
        tracer = TransformerTracer(model)
        tracer.preload_attn_func()

        distributed_option = DistributedOption()
        train_iter_factory = cls.build_iter_factory(
            args=args,
            distributed_option=distributed_option,
            mode="train",
        )
        iterator = train_iter_factory.build_iter(0)

        for iiter, (_, batch) in enumerate(tqdm(iterator)):
            batch = to_device(batch, "cuda" if args.ngpu > 0 else "cpu")
            retval = model(**batch)
            #   a. dict type
            if isinstance(retval, dict):
                loss = retval["loss"]
            else:
                loss, _, _ = retval

            loss.backward()
            tracer.update_accumulator()

            if iiter % 1000 == 999:
                tracer.dump(args.path_head_grad)

        tracer.dump(args.path_head_grad)
