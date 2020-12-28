from django.db import models
import uuid
# Create your models here.


class SearchPhrase(models.Model):
    phrase = models.CharField(max_length=200, help_text="Enter a version of phrase")
    answers = models.TextField()
    id = models.AutoField(primary_key=True)

    def __str__(self):
        return self.phrase
