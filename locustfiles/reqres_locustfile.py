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
        self.client.get("/api/users/2")

    @task(1)
    def create_user(self):
        self.client.post(
            "/api/users",
            json={"name": "morpheus", "job": "leader"},
            headers={"Content-type": "application/json"},
        )

