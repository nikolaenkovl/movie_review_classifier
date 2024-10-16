from django.db import models

class Review(models.Model):
    text = models.TextField()
    rating = models.FloatField()
    status = models.CharField(max_length=10)

    def __str__(self):
        return f'{self.text[:50]}... | {self.status} | {self.rating}'
