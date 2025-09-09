from locust import HttpUser, task, between


class JsonPlaceholderUser(HttpUser):
    # Public fake API
    host = "https://jsonplaceholder.typicode.com"
    wait_time = between(0.5, 1.5)

    @task(3)
    def list_posts(self):
        self.client.get("/posts")

    @task(2)
    def get_post(self):
        self.client.get("/posts/1")

    @task(2)
    def post_comments(self):
        self.client.get("/comments", params={"postId": 1})

    @task(1)
    def create_post(self):
        self.client.post(
            "/posts",
            json={"title": "foo", "body": "bar", "userId": 1},
            headers={"Content-type": "application/json; charset=UTF-8"},
        )

