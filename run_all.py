from pathlib import Path
import sys, subprocess
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.part_a import run as part_a
from src.part_b import run as part_b
from src.part_c import run as part_c
from src.part_d import run as part_d
if __name__=='__main__':
    part_a(); part_b(); part_c(); part_d()
