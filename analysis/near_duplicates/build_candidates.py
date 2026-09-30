"""ND-CW CLI compatibility entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reproducibility/aegis_f1"))
from ndcw.cli import main

if __name__ == "__main__":
    main(['build-candidates'] + sys.argv[1:])
