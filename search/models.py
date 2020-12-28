from django.db import models

# Create your models here.
from django.urls import reverse


class SearchPhrase(models.Model):
    search_phrase = models.CharField(max_length=200, help_text="Enter a version of phrase")
    answers = models.TextField()

    def __str__(self):
        return self.search_phrase

