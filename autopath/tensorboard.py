"""autopath.tensorboard -- Launch TensorBoard for autopath training logs.

Usage::

    autopath.tensorboard [--port PORT] [--logdir DIR]

Defaults:
    PORT    = 7008
    LOGDIR  = ~/autopath/tensorboard

To access from a remote machine::

    ssh -L 7008:127.0.0.1:7008 <hostname>
    then open http://localhost:7008
"""

import argparse
import os
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard for autopath training logs.",
    )
    parser.add_argument(
        "--port", type=int, default=7008,
        help="TensorBoard port (default: 7008)",
    )
    parser.add_argument(
        "--logdir", type=str,
        default=os.path.join(os.environ["HOME"], "autopath", "tensorboard"),
        help="Log directory (default: ~/autopath/tensorboard)",
    )
    parser.add_argument(
        "--foreground", action="store_true",
        help="Run TensorBoard in the foreground (default: background with nohup)",
    )
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    if shutil.which("tensorboard") is None:
        print("Error: tensorboard not found. Install it with:")
        print("  pip install tensorboard")
        raise SystemExit(1)

    print(f"Starting TensorBoard...")
    print(f"  Log directory : {args.logdir}")
    print(f"  Port          : {args.port}")
    print()
    print(f"Access locally  : http://localhost:{args.port}")
    print(f"Access remotely : ssh -L {args.port}:127.0.0.1:{args.port} <hostname>")
    print()

    cmd = [
        "tensorboard",
        f"--logdir={args.logdir}",
        f"--port={args.port}",
    ]

    if args.foreground:
        os.execvp(cmd[0], cmd)
    else:
        outpath = os.path.join(args.logdir, "tensorboard.out")
        pidpath = os.path.join(args.logdir, "tensorboard.pid")
        with open(outpath, "w") as outfile:
            proc = subprocess.Popen(
                cmd,
                stdout=outfile,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"TensorBoard started (PID {proc.pid})")
        print(f"Output log: {outpath}")
        with open(pidpath, "w") as f:
            f.write(str(proc.pid))


if __name__ == "__main__":
    main()
