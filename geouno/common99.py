"""Shared 99-epoch training for GeoUNO and the six Table 1 baselines.

Use ``--model geouno --bank PATH`` for GeoUNO. This entry point stops after
the 99-epoch comparison and evaluates the validation-selected checkpoint.
"""

from .baselines.train import main


if __name__ == "__main__":
    main()
