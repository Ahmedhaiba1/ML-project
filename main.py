
from pathlib import Path
import subprocess
import sys

def main():
    project_dir = Path(__file__).resolve().parent
    eda_script = project_dir / "outputs" / "eda.py"

    subprocess.run(
        [sys.executable, str(eda_script)],
        cwd=project_dir,
        check=True
    )

if __name__ == "__main__":
    main()
