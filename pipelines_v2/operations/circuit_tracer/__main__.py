"""Explicit asset preparation; traces use the ordinary Xenon workflow CLI."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare-modal'])
    parser.parse_args()
    subprocess.run([sys.executable,str(Path(__file__).with_name('_modal_prepare.py'))],check=True)

if __name__=='__main__':main()
