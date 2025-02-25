import logging

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIClient:
    """Handles HTTP requests to the ROI Project Planner API."""

    def __init__(self, base_url="http://localhost:8080/api/v1"):
        self.base_url = base_url

    def create_projects(self, projects):
        """Sends a POST request to create projects."""
        url = f"{self.base_url}/projects"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=projects, headers=headers)
            response.raise_for_status()
            logger.info(f"Projects created successfully: {response.json()}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating projects: {e}")
            return None

    def get_projects(self):
        """Fetches all projects from the API."""
        url = f"{self.base_url}/projects"
        try:
            response = requests.get(url)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("data", [])  # Extracts the list of projects
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching projects: {e}")
            return []

    def maximize_capital(self, max_projects, initial_capital):
        """Sends a POST request to select the best projects based on capital."""
        url = f"{self.base_url}/capital/maximization"
        headers = {"Content-Type": "application/json"}
        payload = {"maxProjects": max_projects, "initialCapital": str(initial_capital)}

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(f"Optimal projects selected: {response.json()}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error selecting projects: {e}")
            return None
