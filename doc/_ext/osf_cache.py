"""The osf command used when building the documentation.

Supports ``osf -p <project> fetch <remote>``. Each file is downloaded once per
build from the URLs conf.py listed in ``$OSF_CACHE_DIR/<project>.json`` and
copied to the current directory.
"""

import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

# Seconds allowed for downloading a file
DOWNLOAD_TIME = 600


def main(args):
    if len(args) != 4 or args[0] != "-p" or args[2] != "fetch":
        sys.exit(f"Unsupported command: osf {' '.join(args)}")
    project, remote = args[1], args[3]

    cache = Path(os.environ["OSF_CACHE_DIR"])
    cached = cache / project / remote
    if not cached.exists():
        urls = json.loads((cache / f"{project}.json").read_text())
        if remote not in urls:
            sys.exit(f"{remote} not found in OSF project {project}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        partial = cached.with_name(cached.name + ".partial")
        deadline = time.time() + DOWNLOAD_TIME
        with urllib.request.urlopen(urls[remote], timeout=60) as response:
            with open(partial, "wb") as file:
                while chunk := response.read(1 << 20):
                    if time.time() > deadline:
                        sys.exit(f"Downloading {remote} from OSF took too long")
                    file.write(chunk)
        partial.rename(cached)

    shutil.copyfile(cached, Path(remote).name)


if __name__ == "__main__":
    main(sys.argv[1:])
