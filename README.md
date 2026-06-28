# World Cup 2026 Prediction Engine

This project is a Python-based sports prediction engine designed to simulate the FIFA World Cup 2026 tournament. It leverages historical match data to train statistical models and performs Monte Carlo simulations to predict tournament outcomes, including group stage standings and knockout phase progression.

## Key Features

- **Statistical Modeling**: Implementation of the Dixon-Coles model for match outcome parameter estimation.
- **Tournament Simulation**: Monte Carlo simulation engine covering group stages and knockout rounds.
- **Data-Driven**: Pipeline for loading and preprocessing historical match data.
- **Configurable**: Centralized hyperparameter and configuration management.

## Project Structure

- `src/`: Core logic
  - `src/models/`: Statistical models (e.g., Dixon-Coles).
  - `src/simulation/`: Tournament simulation logic.
  - `src/data/`: Data loading and preprocessing.
  - `src/config/`: Configuration and hyperparameters.
- `data/`: Datasets, including historical match results.
- `notebooks/`: Exploratory analysis and model development.
- `tests/`: Pytest suite for code validation.

## Getting Started

### Prerequisites

- Python 3.x
- A local virtual environment (`venv/`) is recommended.

### Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup environment variables:
   Ensure the project root is in your `PYTHONPATH` to resolve internal imports:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

## Running Tests

Run the test suite using `pytest`:

```bash
PYTHONPATH=$(pwd) pytest tests/
```
