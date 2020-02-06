"""
Final project for 2019 Tampere University recommender systems course.
Tuomas Luojus & Toni Kuikka.

Program is designed to be used with movielens-dataset, but will work with any
other similarily structured csv dataset file.

Generates five recommendations for a given user and 
answers why-not questions about the recommendations

Run with python manage.py runserver
"""

#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recsite.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
