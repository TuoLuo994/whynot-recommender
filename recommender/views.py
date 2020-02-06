"""
Final project for 2019 Tampere University recommender systems course.
Tuomas Luojus & Toni Kuikka.

Program is designed to be used with movielens-dataset, but will work with any
other similarily structured csv dataset file.

Generates five recommendations for a given user and 
answers why-not questions about the recommendations

Run with python manage.py runserver
"""

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from .forms import WhyNotForm

from . import CoFi
import pandas as pd 
import numpy as np
import sklearn as sk
import statistics
import os

def index(request):
    five_users = []
    for i in range(1, 611):
      five_users.append(i)
    template = loader.get_template('recommender/index.html')
    context = {
        'five_users': five_users,
    }
    return HttpResponse(template.render(context, request))

def recommendations(request, user):

    if request.method == 'POST': # If the form has been submitted
        form = WhyNotForm(request.POST) # A form bound to the POST data
    else:
        form = WhyNotForm() # An unbound form

    for u in range(611):
        if u!=int(user):
            try:
                os.remove(str(u) + '_predicted.csv')
                print("removed " + str(u) + '_predicted.csv')
            except:
                pass

    try:
        predicted_movies = pd.read_csv(user + '_predicted.csv')
    except:
        predicted_movies = CoFi.User_item_score1(int(user))
        predicted_movies.to_csv(user + '_predicted.csv')

    Movie_seen_by_user = CoFi.GetMovieSeen(int(user))
    predicted_movies_5 = predicted_movies.head(5)
    Movie_Names = predicted_movies_5.title.values.tolist()
    template = loader.get_template('recommender/recommendations.html')
    if form.is_valid(): 
        whynot = CoFi.WhyNot(form.cleaned_data['query'], predicted_movies, Movie_seen_by_user, int(user))
    else:
        whynot = ""
    context = {
        'user': user,
        'Movie_Names': Movie_Names,
        'form': form,
        'whynot': whynot,
    }
    return HttpResponse(template.render(context, request))
