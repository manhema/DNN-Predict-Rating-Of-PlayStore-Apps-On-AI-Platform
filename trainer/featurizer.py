import tensorflow as tf

# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [
    # Continuous base columns.
    tf.feature_column.numeric_column('Rating'),
    tf.feature_column.numeric_column('Reviews'),
    tf.feature_column.numeric_column('Size'),
    tf.feature_column.numeric_column('Installs'),
    tf.feature_column.numeric_column('Price'),
    # tf.feature_column.numeric_column('Android_Ver'),

    # For categorical columns with known values we can provide lists
    # of values ahead of time.

    # Categorical base columns
    tf.feature_column.categorical_column_with_vocabulary_list(
        'Category', ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY', 'BOOKS_AND_REFERENCE', 'BUSINESS',
        'COMICS', 'COMMUNICATION', 'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE', 'FOOD_AND_DRINK',
        'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME', 'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',
        'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL', 'TOOLS', 'PERSONALIZATION',
        'PRODUCTIVITY', 'PARENTING', 'WEATHER', 'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'Type', ['Free', 'Paid']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'Content_Rating', ['Everyone', 'Teen', 'Everyone 10+', 'Mature 17+', 'Adults only 18+', 'Unrated']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'Android_Ver', ['4.0', '2.0', '5.0', '3.0', '7.0', '1.0', '6.0', '8.0']),
]


def get_dnn_columns(embedding_size=8):
    (Rating, Reviews, Size, Installs, Price, Category, Type, Content_Rating, Android_Ver) = INPUT_COLUMNS

    deep_columns = [
        # Use indicator columns for low dimensional vocabularies
        tf.feature_column.indicator_column(Type),
        tf.feature_column.indicator_column(Content_Rating),
        tf.feature_column.indicator_column(Android_Ver),
        # Use embedding columns for high dimensional vocabularies
        tf.feature_column.embedding_column(
            Category, dimension=embedding_size),
        Reviews,
        Size,
        Installs,
        Price
    ]

    return deep_columns
