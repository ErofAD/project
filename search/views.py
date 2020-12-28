from django.shortcuts import render

# Create your views here.
from .models import SearchPhrase

from .model.model import Model


def index(request):
    # search_phrase = SearchPhrase.objects.all().count()
    # phrase_version = PhraseVersion.objects.all().count()
    questions = []
    if request.GET.get('search'):
        search = request.GET.get('search')
        model = Model()
        model.init_params()
        questions = model.searching(search)
        new_search_phrase = SearchPhrase(phrase=search, answers=questions)
        new_search_phrase.save()

        # if search in SearchPhrase.objects.all():
        #     questions.append(PhraseVersion.objects.all())
        # else:
        #     questions.append(search)

    return render(
        request,
        'index.html',
        context={'questions': questions}
    )

