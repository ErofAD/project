from django.db import models

# Create your models here.
from django.urls import reverse


# class PhraseVersion(models.Model):
#     version = models.CharField(max_length=200, help_text="Enter a version of phrase")
#
#     def __str__(self):
#         return self.version
#
#
# class SearchPhrase(models.Model):
#     phrase = models.CharField(
#         max_length=200,
#         help_text="Enter a phrase"
#     )
#
#     phrase_version = models.ManyToManyField(PhraseVersion, help_text="Select a true version for this phrase")
#
#     class Meta:
#         ordering = ['phrase']
#
#     def display_phrase(self):
#         return ', '.join([phrase_version.name for phrase_version in self.phrase_version.all()[:5]])
#
#     def get_absolute_url(self):
#         """Returns the url to access a particular book instance."""
#         return reverse('phrase-detail', args=[str(self.id)])
#
#     def __str__(self):
#         return self.phrase


class SearchPhrase(models.Model):
    search_phrase = models.CharField(max_length=200, help_text="Enter a version of phrase")
    answers = models.TextField()

    def __str__(self):
        return self.search_phrase, self.answers

