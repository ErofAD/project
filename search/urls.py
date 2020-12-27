from django.conf.urls import url
from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
]
# urlpatterns += [
#     url(r'^book/Searching/$', views., name='renew-book-librarian'),
# ]