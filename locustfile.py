from locust import HttpUser, task, between

class QuickstartUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def upload_image(self):
        files = {'image': ('filename', open('f1.jpg', 'rb'), 'image/jpeg')}
        self.client.post("/classify", files=files)
