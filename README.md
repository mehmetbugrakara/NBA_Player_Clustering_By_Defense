# NBA Defensive Shot Clustering Pipeline

## Installation
Clone the repository and install dependencies:
git clone https://github.com/mehmetbugrakara/NBA_Player_Clustering_By_Defense.git
cd NBA_Player_Clustering_By_Defense
python -m venv venv
- macOS/Linux
source venv/bin/activate
- Windows
venv\Scripts\activate
pip install -r requirements.txt

## Usage
Run the pipeline:
python start.py
Adjust `seasons` and `n_clusters` in the `__main__` block as needed.

## Logging
• File: logs/analytic.log (INFO+; rotates every 30 days)
• Console: DEBUG level via custom stream handler
• Injected into every class via metaclass=_Base

## Requirements
See requirements.txt for full details. Key packages include:
• numpy==1.26.4
• pandas==2.2.2
• nba_api==1.4.1
• scikit-learn==1.6.0

## License
This project is released under the MIT License. See LICENSE for details.
