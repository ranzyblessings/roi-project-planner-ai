#!/usr/bin/env python3
import argparse
import logging
import random
from dataclasses import dataclass
from typing import List, Tuple

from ai_model import AIModel
from api_client import APIClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Project:
    """Data class representing a project with its attributes."""
    name: str
    requiredCapital: float
    profit: float


class ProjectSelector:
    """Class to handle project selection and optimization."""

    def __init__(self, max_projects: int, initial_capital: float):
        self.max_projects = max_projects
        self.initial_capital = initial_capital
        self.api_client = APIClient()
        self.ai_model = AIModel()
        self.projects: List[Project] = []

    def generate_sample_projects(self, count: int = 10) -> None:
        """Generate random sample projects and store them via API."""
        self.projects = [
            Project(
                name=f"Project {i + 1}",
                requiredCapital=float(random.randint(0, 200)),
                profit=float(random.randint(50, 500))
            ) for i in range(count)
        ]
        try:
            self.api_client.create_projects([p.__dict__ for p in self.projects])
            logger.info(f"Successfully created {count} sample projects")
        except Exception as e:
            logger.error(f"Failed to create projects: {str(e)}")
            raise

    def fetch_projects(self) -> List[Project]:
        """Fetch projects from API and convert to Project objects, filtering expected fields."""
        try:
            project_data = self.api_client.get_projects()
            if not project_data:
                raise ValueError("No project data returned from API")
            # Filter only the fields Project expects
            self.projects = [
                Project(
                    name=proj["name"],
                    requiredCapital=float(proj["requiredCapital"]),
                    profit=float(proj["profit"])
                ) for proj in project_data
            ]
            return self.projects
        except Exception as e:
            logger.error(f"Failed to fetch projects: {str(e)}")
            raise

    def train_and_predict(self) -> List[int]:
        """Train AI model and predict project selection."""
        capital_values = [p.requiredCapital for p in self.projects]
        profit_values = [p.profit for p in self.projects]
        labels = [1 if p.requiredCapital <= self.initial_capital else 0
                  for p in self.projects]

        try:
            self.ai_model.train(list(zip(capital_values, profit_values)), labels)
            predictions = self.ai_model.predict(list(zip(capital_values, profit_values)))
            if predictions is None:
                raise ValueError("AI model returned None predictions")
            return predictions
        except Exception as e:
            logger.error(f"AI training/prediction failed: {str(e)}")
            raise

    def select_optimal_projects(self) -> Tuple[List[Project], float]:
        """Select optimal projects based on predictions and constraints."""
        predictions = self.train_and_predict()
        candidates = [p for i, p in enumerate(self.projects) if predictions[i] == 1]

        candidates.sort(key=lambda p: p.profit / max(p.requiredCapital, 1), reverse=True)

        selected = []
        remaining_capital = self.initial_capital

        for project in candidates:
            if (len(selected) < self.max_projects and
                    project.requiredCapital <= remaining_capital):
                selected.append(project)
                remaining_capital += project.profit

        return selected, remaining_capital

    def display_projects(self, projects: List[Project], title: str) -> None:
        """Display projects in a formatted manner."""
        print(f"\n=== {title} ===")
        for p in projects:
            print(f"{p.name:10} | Capital: ${p.requiredCapital:6.2f} | Profit: ${p.profit:6.2f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with validation."""
    parser = argparse.ArgumentParser(description="Project Selection for Maximizing Capital")
    parser.add_argument(
        '--maxProjects',
        type=int,
        default=5,
        help='Maximum number of projects to select (must be positive)'
    )
    parser.add_argument(
        '--initialCapital',
        type=float,
        default=100.0,
        help='Initial capital to start with (must be non-negative)'
    )
    args = parser.parse_args()

    if args.maxProjects <= 0:
        parser.error("maxProjects must be positive")
    if args.initialCapital < 0:
        parser.error("initialCapital must be non-negative")

    return args


def main():
    """Main execution flow."""
    args = parse_args()

    try:
        selector = ProjectSelector(args.maxProjects, args.initialCapital)

        # This is used for testing purposes to generate random test data.
        selector.generate_sample_projects()
        selector.display_projects(selector.projects, "Created Projects")

        fetched_projects = selector.fetch_projects()
        selector.display_projects(fetched_projects, "Fetched Projects")

        selected_projects, remaining_capital = selector.select_optimal_projects()

        print(f"\n=== Selected Projects for Capital Maximization ===")
        print(f"Max Projects: {args.maxProjects} | Initial Capital: ${args.initialCapital:.2f}")
        selector.display_projects(selected_projects, "Selected Projects")
        print(f"\nRemaining Capital: ${remaining_capital:.2f}")

    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
