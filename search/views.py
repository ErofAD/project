from django.shortcuts import render

# Create your views here.
from .models import PhraseVersion, SearchPhrase

from .model.model import Search


def index(request):
    search_phrase = SearchPhrase.objects.all().count()
    phrase_version = PhraseVersion.objects.all().count()
    questions = []
    if request.GET.get('search'):
        search = request.GET.get('search')
        questions = Search.searching(search)
        # if search in SearchPhrase.objects.all():
        #     questions.append(PhraseVersion.objects.all())
        # else:
        #     questions.append(search)

    return render(
        request,
        'index.html',
        context={'search_phrase': search_phrase, 'phrase_version': phrase_version, 'questions': questions}
    )

