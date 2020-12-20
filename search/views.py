from django.shortcuts import render

# Create your views here.
from .models import PhraseVersion, SearchPhrase

def index(request):
    searchPhrase = SearchPhrase.objects.all().count()
    phraseVersion = PhraseVersion.objects.all().count()
    return render(
        request,
        'index.html',
        context={'searchPhrase': searchPhrase, 'phraseVersion': phraseVersion}
    )