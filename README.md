# ROI Project Planner AI

The **AI-powered ROI Project Planner** uses machine learning to select the most profitable projects within capital
constraints. It sends requests to the ROI Project Planner backend for further processing.

To support AI functions, we utilize **TensorFlow**, **Pandas**, **NumPy**, and **Scikit-learn** for tasks such as data
processing, machine learning model development, and performance optimization.

## How to Set Up & Run Locally

### Prerequisites

- [ROI Project Planner (Backend)](https://github.com/ranzyblessings/roi-project-planner) - must be up and running
  locally
- [Python v3.12.7](https://www.python.org/) (or latest)
- [Poetry v2.1](https://python-poetry.org/docs/) (or latest)

### Install Dependencies

```bash
poetry install
```

**Run the Project**

```bash
poetry run python src/main.py
```

**Run Tests**

```bash
PYTHONPATH=. poetry run pytest
```