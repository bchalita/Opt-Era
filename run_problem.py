"""
Generic runner for OptiMUS on a NEW problem instance (not necessarily in any DB).

Goal:
- You provide a problem folder containing:
  - desc.txt
  - params.json
  - labels.json (optional but recommended for some RAG modes)
- This script runs main.py end-to-end and (optionally) exports a LaTeX report.

This is NOT a custom solver for one problem. It's a reusable harness for ANY problem you describe.

Usage:
  source .venv/bin/activate
  export OPENAI_API_KEY="..."
  python run_problem.py --problem-dir /path/to/problem_dir --rag-mode problem_description --model gpt-5 --export-tex 1
"""

import argparse
import os
import subprocess


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--problem-dir", type=str, required=True, help="Directory containing desc.txt/params.json[/labels.json]")
    p.add_argument("--rag-mode", type=str, default="problem_description", choices=["problem_description", "constraint_or_objective", "problem_labels", "none"])
    p.add_argument("--model", type=str, default="gpt-5")
    p.add_argument("--devmode", type=int, default=1)
    p.add_argument("--export-tex", type=int, default=1)
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export OPENAI_API_KEY before running.")

    repo_root = os.path.abspath(os.path.dirname(__file__))
    problem_dir = os.path.abspath(args.problem_dir)

    cmd = [
        "python",
        os.path.join(repo_root, "main.py"),
        "--dir",
        problem_dir,
        "--devmode",
        str(args.devmode),
        "--model",
        args.model,
        "--export-tex",
        str(args.export_tex),
    ]
    if args.rag_mode != "none":
        cmd += ["--rag-mode", args.rag_mode]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root)

    run_dir = os.path.join(problem_dir, "run_dev" if args.devmode else "run_*")
    print("\nDone. Look in:")
    print(f"- {run_dir}")
    if args.export_tex:
        print(f"- {os.path.join(problem_dir, 'run_dev', 'report.tex')}")


if __name__ == "__main__":
    main()

