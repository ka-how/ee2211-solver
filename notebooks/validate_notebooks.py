"""Validate and execute all notebooks with the current Python interpreter."""

import ast
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    folder = Path(__file__).resolve().parent
    notebooks = sorted(folder.glob("*.ipynb"))
    if len(notebooks) != 9:
        raise RuntimeError(f"Expected nine tab notebooks, found {len(notebooks)}.")
    print(f"Executing all notebooks with Python {sys.version.split()[0]}", flush=True)
    with tempfile.TemporaryDirectory(prefix="ee2211-notebook-check-") as temporary:
        environment = dict(os.environ,
                           EE2211_NOTEBOOK_CACHE=str(Path(temporary) / "cache.sqlite3"),
                           IPYTHONDIR=str(Path(temporary) / "ipython"))
        for path in notebooks:
            document = nbformat.read(path, as_version=4)
            nbformat.validate(document)
            for index, cell in enumerate(document.cells):
                if cell.cell_type == "code":
                    ast.parse(cell.source, filename=f"{path.name}:cell{index}", feature_version=(3, 11))
            manager = KernelManager(kernel_name="python3")
            # Choose this interpreter even if another environment owns python3's kernelspec.
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            client = NotebookClient(document, km=manager, timeout=120,
                                    resources={"metadata": {"path": str(folder)}})
            print(f"Running {path.name} ...", flush=True)
            # A supplied manager is caller-owned, so explicitly clean it up on errors too.
            try:
                client.execute(env=environment)
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
            print(f"PASS {path.name}", flush=True)
    print("All nine notebooks passed; notebook files and your cache were not changed.")


if __name__ == "__main__":
    main()
