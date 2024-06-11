import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

# Title of the application
st.title('Anime Recommendation System')

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('data/animes_update_base.csv')

    data['year'] = pd.to_numeric(data['year'], errors='coerce').fillna(0).astype(int)
    data['episodes'] = pd.to_numeric(data['episodes'], errors='coerce').fillna(0).astype(int)

    return data

# Nettoyage des données
anime_data = load_data()

# Preprocess data and compute TF-IDF
def preprocess_and_compute_features(data):
    selected_columns = ['title','synopsis', 'genres', 'year', 'producers', 'studios', 'themes']  # Add additional columns
    anime_features = data[selected_columns]
    anime_features['synopsis'] = anime_features['synopsis'].fillna('')
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(anime_features['synopsis'])
    
    # Initialiser le binariseur pour les genres, thèmes et sources
    genre_binarizer = MultiLabelBinarizer()
    theme_binarizer = MultiLabelBinarizer()

    # Encoder les genres, thèmes et sources
    genres_binary = genre_binarizer.fit_transform(anime_features['genres'].apply(ast.literal_eval))
    themes_binary = theme_binarizer.fit_transform(anime_features['themes'].apply(ast.literal_eval))

    # Concaténer les matrices TF-IDF et les matrices binaires pour former le vecteur de caractéristiques complet
    feature_matrix = hstack([tfidf_matrix, genres_binary, themes_binary])

    return feature_matrix, tfidf_vectorizer

feature_matrix, tfidf_vectorizer = preprocess_and_compute_features(anime_data)

# Sidebar for user inputs
with st.sidebar:
    st.header('User Inputs')
    anime_list = anime_data['title'].tolist()
    selected_anime = st.selectbox("Select an Anime", anime_list)
    num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

# Get recommendations based on cosine similarity
def get_recommendations(selected_anime, data, feature_matrix, num_recs):
    # Trouver l'index de l'anime sélectionné dans le DataFrame original
    idx = data.index[data['title'] == selected_anime].tolist()[0]

    # Convertir la matrice de caractéristiques au format CSR pour l'indexation
    feature_matrix_csr = feature_matrix.tocsr()

    # Utiliser l'index pour accéder à la rangée correspondante dans la matrice CSR
    feature_idx = feature_matrix_csr[idx]

    # Calculer les similarités
    cosine_similarities = cosine_similarity(feature_idx, feature_matrix_csr).flatten()
    similar_indices = np.argsort(-cosine_similarities)[1:num_recs+1]

    # Retourner les animes recommandés en utilisant les indices du DataFrame original
    recommended_animes = data.iloc[similar_indices]
    return recommended_animes


# Displaying recommendations with additional information
if st.button("Get Recommendations"):
    st.header("Anime Recommendations")
    recommendations = get_recommendations(selected_anime, anime_data, feature_matrix, num_recommendations)
    for _, rec in recommendations.iterrows():
        col1, col2, col3 = st.columns([3, 2, 2])  # Ajustement de la répartition des colonnes
        

        with col1:
            #st.image(rec['image_url'], use_column_width=True)
            # Assume `rec['images']` is a string that looks like a dictionary
            images_dict_str = rec['images']  # This is a string, not a dictionary
            images_dict = ast.literal_eval(images_dict_str)  # Convert string to dictionary

            # Now you can access the URLs in the dictionary
            large_image_url = images_dict['jpg']['large_image_url']
            st.image(large_image_url, use_column_width=True)
            print(rec['images'])

        with col2:
            st.markdown("#### Titre")
            st.write(rec['title'])
            st.markdown('#### Genres')
            genres_list = ast.literal_eval(rec['genres'])
            genres_str = ', '.join(genres_list)
            st.write(genres_str)
            st.markdown('#### Thèmes')
            themes_list = ast.literal_eval(rec['themes'])
            themes_str = ', '.join(themes_list)
            st.write(themes_str)
            st.markdown('#### Statut')
            st.write(rec['status'])
            

        with col3:
            st.markdown("#### Score")
            st.write(rec['score'])
            st.markdown("#### Épisodes")
            st.write(rec['episodes'])
            st.markdown("#### Année")        
            st.write(rec['year'])
            st.markdown('#### Type')
            st.write(rec['type'])
            

        st.write("Synopsis:", rec['synopsis'])  # Affichage du synopsis en dessous des informations
        producers_list = ast.literal_eval(rec['producers'])
        producers_str = ', '.join(producers_list)
        st.write("Producers:", producers_str)  # Affichage des producteurs 

        st.markdown("#### Studios")
        studios_list = ast.literal_eval(rec['studios'])
        studios_str = ', '.join(studios_list)
        st.write(studios_str)

        # Traitement et affichage des thèmes
        theme_dict = ast.literal_eval(rec['theme'])
        openings = theme_dict.get('openings', [])
        endings = theme_dict.get('endings', [])
        openings_str = ', '.join(openings) if openings else 'Aucun'
        endings_str = ', '.join(endings) if endings else 'Aucun'
        st.markdown("#### Openings")
        st.write(openings_str)
        st.markdown("#### Endings")
        st.write(endings_str)

        st.write("---")  # Séparateur pour chaque recommandation

# Additional Information
st.sidebar.header("Informations")
st.sidebar.info("This is a simple anime recommendation system. Select an anime from the dropdown and choose the number of recommendations you want. Enjoy ! UwU")
