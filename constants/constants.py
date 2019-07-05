#!/bin/bash

CSV_COLUMNS = [
    'App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type',
    'Price', 'Content_Rating', 'Genres', 'Last Updated', 'Current Ver',
    'Android_Ver'
]

CSV_COLUMN_DEFAULTS = [[''], [''], [0], [0], [0.0], [0], [''], [0.0], [''],
                       [''], [''], [''], ['']]

LABEL_COLUMN = 'Rating'

# LABELS = ['Low', 'High']
LABELS = [0, 1]