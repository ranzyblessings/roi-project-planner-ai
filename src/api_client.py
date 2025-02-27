import logging
from typing import List, Dict, Any, Optional

from requests import Session, RequestException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class APIConfig:
    """Configuration class for API client settings."""
    DEFAULT_BASE_URL = "http://localhost:8080/api/v1"
    HEADERS = {"Content-Type": "application/json"}
    TIMEOUT = 10  # seconds


class APIClient:
    """
    A robust HTTP client for interacting with the ROI Project Planner API.

    Attributes:
        base_url: Base URL for API endpoints.
        session: Persistent HTTP session for connection pooling.
    """

    def __init__(self, base_url: str = APIConfig.DEFAULT_BASE_URL, session: Optional[Session] = None) -> None:
        """
        Initialize the API client with a base URL and optional session.

        Args:
            base_url: Base URL for the API (default: localhost).
            session: Optional requests Session for dependency injection (default: new Session).
        """
        self.base_url = base_url.rstrip("/")  # Ensure no trailing slash
        self.session = session if session is not None else Session()
        self.session.headers.update(APIConfig.HEADERS)
        logger.debug(f"APIClient initialized with base_url: {self.base_url}")

    def _request(self, method: str, endpoint: str, json: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generic method to handle HTTP requests with error handling.

        Args:
            method: HTTP method (e.g., 'GET', 'POST').
            endpoint: API endpoint path (relative to base_url).
            json: Payload for POST requests.

        Returns:
            JSON response or None if the request fails.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json,
                timeout=APIConfig.TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"API request failed: {method} {url} - {str(e)}")
            return None

    def create_projects(self, projects: List[Dict[str, Any]]) -> Optional[Dict]:
        """
        Create projects via POST request.

        Args:
            projects: List of project dictionaries to create.

        Returns:
            API response JSON or None if the request fails.
        """
        response = self._request("POST", "/projects", json=projects)
        if response is not None:
            logger.info(f"Projects created successfully: {response}")
        return response

    def get_projects(self) -> List[Dict[str, Any]]:
        """
        Fetch all projects from the API.

        Returns:
            List of project dictionaries, empty list if the request fails.
        """
        response = self._request("GET", "/projects")
        if response is not None:
            projects = response.get("data", [])
            logger.info(f"Fetched {len(projects)} projects.")
            return projects
        return []

    def maximize_capital(self, max_projects: int, initial_capital: float) -> Optional[Dict]:
        """
        Select optimal projects based on capital constraints.

        Args:
            max_projects: Maximum number of projects to select.
            initial_capital: Available initial capital.

        Returns:
            API response JSON or None if the request fails.
        """
        payload = {
            "maxProjects": max_projects,
            "initialCapital": str(initial_capital)  # API expects string
        }
        response = self._request("POST", "/capital/maximization", json=payload)
        if response is not None:
            logger.info(f"Optimal projects selected: {response}")
        return response

    def close(self) -> None:
        """Close the session to free resources."""
        self.session.close()
        logger.debug("APIClient session closed.")
