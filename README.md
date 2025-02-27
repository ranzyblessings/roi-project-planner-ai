# ROI Project Planner AI

The **AI-powered ROI** Project Planner leverages machine learning to identify the most profitable projects within
capital constraints and integrates with the **ROI Project Planner (Backend)** to query project data.

To support its AI capabilities, we use **TensorFlow**, **Pandas**, **NumPy**, and **scikit-learn** for tasks like data
processing, model development, and performance optimization. Specifically, we train a TensorFlow model to optimize
project selection, enhancing decision-making based on historical data and financial constraints.

_The problem statement is adapted from the [LeetCode IPO problem](https://leetcode.com/problems/ipo), with modifications
tailored to meet our requirements._

[![Run Tests](https://github.com/ranzyblessings/roi-project-planner-ai/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/ranzyblessings/roi-project-planner-ai/actions/workflows/build-and-test.yaml)

## How to Set Up and Run Locally

### Prerequisites

- [ROI Project Planner (Backend)](https://github.com/ranzyblessings/roi-project-planner) - Must be installed and running
  locally.
- [Python v3.12.7](https://www.python.org/) (or the latest version).
- [Poetry v1.8.3](https://python-poetry.org/docs/) (or the latest version).

### Installation

Set up the project by installing its dependencies, including TensorFlow, Pandas, NumPy, scikit-learn, and others, using
Poetry.

```bash
poetry install
```

**Running the Project**

Run the project to analyze and prioritize projects based on ROI within specified constraints.

```bash
poetry run python src/main.py --maxProjects 5 --initialCapital 200
```

- `--maxProjects`: Limits the number of projects to select (e.g., 5).
- `--initialCapital`: Sets the initial capital budget (e.g., $200).

The script employs a trained machine learning model to optimize project selection, drawing on historical data and
financial constraints.

By default, it generates 10 random projects, integrating with the ROI Project Planner (Backend),
to help the AI learn from a large dataset and refine its performance under varying conditions.

**Running Tests**

Validate the project’s functionality by running the test suite.

```bash
PYTHONPATH=. poetry run pytest
```
