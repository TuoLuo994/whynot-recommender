from django import forms

class WhyNotForm(forms.Form):
    query = forms.CharField(label='Ask a why-not -question about your search result:', max_length=150, required=False)
