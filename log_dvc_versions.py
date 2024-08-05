import subprocess


def log_dvc_versions():
    result = subprocess.run(["dvc", "repro"], capture_output=True, text=True)
    with open("dvc_logs.md", "w") as f:
        f.write("## DVC Repro Output\n")
        f.write(result.stdout)
        f.write("\n")

    result = subprocess.run(["dvc", "log"], capture_output=True, text=True)
    with open("dvc_logs.md", "a") as f:
        f.write("## DVC Log\n")
        f.write(result.stdout)
        f.write("\n")


if __name__ == "__main__":
    log_dvc_versions()
