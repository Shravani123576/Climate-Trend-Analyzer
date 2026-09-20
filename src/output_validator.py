from pathlib import Path

def validate_outputs():

    required = [
        "outputs/graphs",
        "outputs/tables",
        "reports"
    ]

    for folder in required:

        if not Path(folder).exists():

            raise FileNotFoundError(
                f"{folder} missing."
            )

    print("Output validation passed.")