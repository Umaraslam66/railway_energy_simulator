# Railway Energy Efficiency Simulator 🚂

A Python-based simulation tool for analyzing and optimizing energy consumption in railway operations. This project combines physics-based modeling with machine learning to provide insights into train energy efficiency.

## Features

- **Energy Consumption Simulation**
  - Physics-based train motion modeling
  - Support for both electric and diesel trains
  - Consideration of track gradients and speed limits
  - Regenerative braking simulation

- **Machine Learning Analysis**
  - Energy consumption prediction (97.5% test accuracy)
  - Pattern analysis using clustering
  - Optimization recommendations
  - Performance comparison between train types

- **Interactive Visualizations**
  - Energy consumption dashboards
  - Speed profiles
  - Efficiency comparisons
  - Pattern analysis plots

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/railway_energy_simulator.git
cd railway_energy_simulator

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
railway_energy_simulator/
├── src/
│   ├── core/
│   │   ├── train.py         # Train parameters and physics
│   │   ├── track.py         # Track segment definitions
│   │   └── simulator.py     # Core simulation engine
│   ├── ml/
│   │   ├── predictor.py     # Energy prediction models
│   │   ├── optimizer.py     # Schedule optimization
│   │   └── patterns.py      # Pattern analysis
│   └── visualization/
│       └── dashboards.py    # Interactive visualizations
├── data/
│   ├── sample/             # Sample data files
│   └── processed/          # Processed simulation results
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── main.py               # Main execution script
```

## Usage

Run the main simulation:
```bash
python main.py
```

This will:
1. Run simulations for both electric and diesel trains
2. Perform ML analysis
3. Generate visualizations
4. Save results in the `output` directory

## Example Output

The simulator generates several outputs:
- Energy consumption profiles
- Speed vs. efficiency analysis
- Comparative analysis between train types
- Optimization recommendations

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Plotly
- (See requirements.txt for complete list)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built as a learning project for combining physics simulation with ML
- Inspired by real-world railway energy efficiency challenges
- Special thanks to the open-source community for the tools and libraries used

## Author

[Your Name]