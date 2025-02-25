import logging
import random

from ai_model import AIModel
from api_client import APIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User-defined parameters
max_projects = 5
initial_capital = 100.00

# Initialize API client and AI model
api_client = APIClient()
ai_model = AIModel()

# Create 10 random projects
projects = [
    {"name": f"Project {i + 1}", "requiredCapital": random.randint(0, 200), "profit": random.randint(50, 500)}
    for i in range(10)
]
api_client.create_projects(projects)

# Display created projects
print("\n=== Created Projects ===")
for project in projects:
    print(f"{project['name']:10} | Capital: ${project['requiredCapital']:3} | Profit: ${project['profit']:3}")

# Fetch projects
project_data = api_client.get_projects()
if not project_data:
    logger.error("Failed to fetch project data. Exiting.")
    exit()

# Prepare data for AI training
capital_values = []
profit_values = []
labels = []  # AI learns from past project selections (1 = selected, 0 = not)

for project in project_data:
    capital_values.append(project["requiredCapital"])
    profit_values.append(project["profit"])
    labels.append(1 if project["profit"] >= 200 else 0)  # Rule-based selection

# Train the AI model
ai_model.train_model(list(zip(capital_values, profit_values)), labels)

# Predict project selection
predictions = ai_model.predict(list(zip(capital_values, profit_values)))
selected_projects = []

if predictions is not None:
    selected_projects = [
        project for i, project in enumerate(project_data) if predictions[i] == 1
    ]

# Limit selected projects based on maxProjects and available capital
selected_projects.sort(key=lambda p: p["profit"], reverse=True)  # Sort by highest profit
final_selection = []
remaining_capital = initial_capital

for project in selected_projects:
    if len(final_selection) < max_projects and project["requiredCapital"] <= remaining_capital:
        final_selection.append(project)
        remaining_capital += project["profit"] - project["requiredCapital"]

# Display selected projects
print("\n=== Selected Projects for Capital Maximization ===")
print(f"Max Projects: {max_projects} | Initial Capital: ${initial_capital:.2f}\n")

for project in final_selection:
    print(f"{project['name']:10} | Capital Required: ${project['requiredCapital']:3} | Profit: ${project['profit']:3}")

print(f"\nUpdated Maximized Capital: ${remaining_capital:.2f}")
