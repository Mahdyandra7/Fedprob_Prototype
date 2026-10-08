"""fedprob: prototipe federated probabilistic forecasting pada data simulasi."""

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

__version__ = "0.1.0"

# Pasangan kuantil -> pita interval (lebar ke sempit): (bawah, atas, label)
BANDS = ((0.05, 0.95, "90%"), (0.10, 0.90, "80%"), (0.25, 0.75, "50%"))
