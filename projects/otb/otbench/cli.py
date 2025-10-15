"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd: str) -> int:
    return os.system(cmd)


def cmd_prepare(args: argparse.Namespace) -> int:
    return _run(f"python {os.path.join(ROOT, 'prepare.py')}")


def cmd_generate(args: argparse.Namespace) -> int:
    parts = [
        f"python {os.path.join(ROOT, 'generate.py')}",
        f"--model {args.model}",
        f"--dataset {args.dataset}",
        f"--output {args.output}",
        f"--multiprocessing {args.multiprocessing}",
        f"--temperature {args.temperature}",
        f"--num_gens {args.num_gens}",
    ]
    if args.enable_nothink:
        parts.append("--enable_nothink")
    if args.run_locally:
        parts.append("--run_locally")
    return _run(" ".join(parts))


def cmd_eval(args: argparse.Namespace) -> int:
    return _run(
        f"python {os.path.join(ROOT, 'eval.py')} {args.input} --model {args.model}"
    )


def cmd_create_overthink(args: argparse.Namespace) -> int:
    return _run(
        f"python {os.path.join(ROOT, 'otb_creation', 'create_overthink.py')} --model_path {args.model}"
    )


def cmd_filter_overthink(args: argparse.Namespace) -> int:
    return _run(f"python {os.path.join(ROOT, 'otb_creation', 'filter_overthink.py')}")


def cmd_create_underthink(args: argparse.Namespace) -> int:
    return _run(f"python {os.path.join(ROOT, 'otb_creation', 'create_underthink.py')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="otbench")
    sub = parser.add_subparsers(dest="command")

    p_prepare = sub.add_parser("prepare")
    p_prepare.set_defaults(func=cmd_prepare)

    p_run = sub.add_parser("generate")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--dataset", default="data/otb_full")
    p_run.add_argument("--output", default="final_outputs/otbench/{{model}}.jsonl")
    p_run.add_argument("--multiprocessing", type=int, default=1)
    p_run.add_argument("--temperature", type=float, default=0.6)
    p_run.add_argument("--num_gens", type=int, default=8)
    p_run.add_argument("--enable_nothink", action="store_true", default=False)
    p_run.add_argument("--run_locally", action="store_true", default=False)
    p_run.set_defaults(func=cmd_generate)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("input")
    p_eval.add_argument("--model", required=True)
    p_eval.set_defaults(func=cmd_eval)

    p_co = sub.add_parser("create_overthink")
    p_co.add_argument(
        "--model",
        dest="model",
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    )
    p_co.set_defaults(func=cmd_create_overthink)

    p_fo = sub.add_parser("filter_overthink")
    p_fo.set_defaults(func=cmd_filter_overthink)

    p_cu = sub.add_parser("create_underthink")
    p_cu.set_defaults(func=cmd_create_underthink)

    def cmd_run_all(args: argparse.Namespace) -> int:
        ret = cmd_prepare(args)
        if ret != 0:
            return ret
        ret = cmd_generate(args)
        if ret != 0:
            return ret
        output = args.output if args.output else "final_outputs/otbench/{{model}}.jsonl"
        output = output.replace("{{model}}", args.model.replace("/", "-"))
        return cmd_eval(argparse.Namespace(input=output, model=args.model))

    p_runall = sub.add_parser("run")
    p_runall.add_argument("--model", required=True)
    p_runall.add_argument("--dataset", default="data/otb_full")
    p_runall.add_argument("--output", default="final_outputs/otbench/{{model}}.jsonl")
    p_runall.add_argument("--multiprocessing", type=int, default=1)
    p_runall.add_argument("--temperature", type=float, default=0.6)
    p_runall.add_argument("--num_gens", type=int, default=8)
    p_runall.add_argument("--enable_nothink", action="store_true", default=False)
    p_runall.add_argument("--run_locally", action="store_true", default=False)
    p_runall.set_defaults(func=cmd_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    code = args.func(args)
    sys.exit(os.WEXITSTATUS(code) if isinstance(code, int) else 0)


if __name__ == "__main__":
    main()
