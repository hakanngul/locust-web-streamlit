import random
from locust import HttpUser, task, between


class ReqResUser(HttpUser):
    # Public fake API
    host = "https://reqres.in"
    wait_time = between(0.5, 1.5)

    @task(3)
    def list_users(self):
        self.client.get("/api/users", params={"page": 2})

    @task(2)
    def get_user(self):
        user_id = random.randint(1, 12)
        self.client.get(f"/api/users/{user_id}")

    @task(1)
    def create_user(self):
        self.client.post(
            "/api/users",
            json={"name": "morpheus", "job": "leader"},
            headers={"Content-type": "application/json"},
        )

    @task(1)
    def update_user_put(self):
        user_id = random.randint(1, 12)
        self.client.put(
            f"/api/users/{user_id}",
            json={"name": "neo", "job": "the one"},
            headers={"Content-type": "application/json"},
        )

    @task(1)
    def update_user_patch(self):
        user_id = random.randint(1, 12)
        self.client.patch(
            f"/api/users/{user_id}",
            json={"job": "zion ops"},
            headers={"Content-type": "application/json"},
        )

    @task(1)
    def delete_user(self):
        user_id = random.randint(1, 12)
        self.client.delete(f"/api/users/{user_id}")

    @task(2)
    def list_resources(self):
        self.client.get("/api/unknown")

    @task(1)
    def get_resource(self):
        res_id = random.randint(1, 12)
        self.client.get(f"/api/unknown/{res_id}")

    @task(1)
    def delayed_users(self):
        # Simulate slow responses (3s delay)
        self.client.get("/api/users", params={"delay": 3})

    @task(1)
    def register_success(self):
        self.client.post(
            "/api/register",
            json={"email": "eve.holt@reqres.in", "password": "pistol"},
            headers={"Content-type": "application/json"},
        )

    @task(1)
    def register_fail(self):
        # Intentionally missing password -> 400
        self.client.post(
            "/api/register",
            json={"email": "sydney@fife"},
            headers={"Content-type": "application/json"},
        )

    @task(1)
    def login_success(self):
        self.client.post(
            "/api/login",
            json={"email": "eve.holt@reqres.in", "password": "cityslicka"},
            headers={"Content-type": "application/json"},
        )

    @task(1)
    def login_fail(self):
        # Missing password -> 400
        self.client.post(
            "/api/login",
            json={"email": "peter@klaven"},
            headers={"Content-type": "application/json"},
        )
