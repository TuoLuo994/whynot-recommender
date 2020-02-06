# whynot-recommender
Simple recommendation system with why-not queries

## Usage

python-libraries needed to run this system: pandas, numpy, sklearn, statistics, django
```bash
pip install pandas
pip install nympy
pip install sklearn
pip install statistics
pip install django
python manage.py runserver
```

* Enter http://127.0.0.1:8000/recommender into your browser address bar
* Choose a user from the list
* After the system calculates recommendations for that user, you can ask why-not questions like:
  -Why was Return of the Jedi not recommended to me?
  -Why was A New Hope recommended whereas Phantom Menace was not? 
  -Why is Raiders of the Lost Ark not in a higher rating?
  -Why is The Empire Strikes Back below Hoop Dreams?
  -Why are there no more Horror movies?
  -Why so few Fantasy movies but so many Sci-Fi movies?

The recommender uses an included movielens-dataset, but will work with any other similarly structured csv-file.
