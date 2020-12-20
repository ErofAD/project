from django.contrib import admin

from .models import PhraseVersion, SearchPhrase

admin.site.register(PhraseVersion)
admin.site.register(SearchPhrase)

