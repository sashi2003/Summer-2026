import streamlit as st
import pandas as pd
import plotly.express as px

#Loading Data
df = pd.read_csv("../pandas/spotify-tracks-dataset.csv")
df = df.drop(columns = ['Unnamed: 0.1', 'Unnamed: 0'])
df = df.dropna()
df['duration_sec'] = df['duration_ms']/1000

st.title("Spotify Music Analysis Dashboard")
st.write("Exploring 114,000 tracks across genres, energy, danceability and popularity.")

st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Total Tracks", "114,000")
col2.metric("Total Genres", df['track_genre'].nunique())
col3.metric("Average Popularity", round(df['popularity'].mean(), 1))

st.sidebar.title("Filters")
genres = df['track_genre'].unique().tolist()
selected_genre = st.sidebar.selectbox("Select a Genre", ["All"] + sorted(genres))
if selected_genre != "All":
    filtered_df = df[df['track_genre'] == selected_genre]
else:
    filtered_df = df
# Chart 1 - Energy by Genre
st.subheader("Highest Energy Genres")

genre_energy = df.groupby('track_genre')['energy'].mean()
genre_energy = genre_energy.sort_values(ascending=False).head(10).reset_index()

fig1 = px.bar(genre_energy,
              x='energy',
              y='track_genre',
              orientation='h',
              title='Top 10 Highest Energy Genres',
              labels={'energy': 'Average Energy', 'track_genre': 'Genre'},
              color='energy',
              color_continuous_scale='reds')

fig1.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig1)

# Chart 2 - Danceability by Genre
st.subheader("Highest Danceable Genres")
genre_danceability = df.groupby('track_genre')['danceability'].mean()
genre_danceability = genre_danceability.sort_values(ascending = False).head(10).reset_index()
fig2 = px.bar(genre_danceability, 
              x = 'danceability', 
              y = 'track_genre', 
              orientation = 'h', 
              title = 'Top 10 Highest Danceable Genres', 
              labels = {'danceability': 'Average Danceability', 'track_genre': 'Genre'}, 
              color = 'danceability', 
              color_continuous_scale = 'blues')

fig2.update_layout(yaxis = {'categoryorder': 'total ascending'})
st.plotly_chart(fig2)

# Chart 3
st.subheader("Energy vs Danceability by Genre")
genre_summary = filtered_df.groupby('track_genre').agg(
    {
        'energy': 'mean', 
        'danceability': 'mean', 
        'popularity': 'mean'
    }
).reset_index()

fig3 = px.scatter(genre_summary, 
                  x = 'energy', 
                  y = 'danceability', 
                  hover_name = 'track_genre', 
                  size = 'popularity', 
                  color = 'popularity', 
                  color_continuous_scale = 'viridis', 
                  title = 'Energy vs Danceability by Genre', 
                  labels = {
                      'energy': 'Average Energy', 
                      'danceability': 'Average Danceability', 
                      'popularity': 'Avg Popularity'
                  })
st.plotly_chart(fig3)

# Chart 4
st.subheader("Audio Features Correlation with Popularity")
features = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo', 'duration_sec']

correlations = filtered_df[features].corrwith(filtered_df['popularity'])
corr_df = correlations.reset_index()
corr_df.columns = ['feature', 'correlation']
corr_df = corr_df.sort_values('correlation', ascending = True)

corr_df['color'] = corr_df['correlation'].apply(lambda x: 'positive' if x>0 else 'negative')
fig4 = px.bar(corr_df, 
             x = 'correlation',
             y = 'feature', 
             orientation = 'h', 
             title = 'Correlations between the features', 
             color = 'color', 
             color_discrete_map = {'positive': 'steelblue', 'negative': 'tomato'}, 
             labels = {
                 'feature': 'feature', 
                 'correlation': 'correlation'
             })

fig4.update_layout(
    legend_title_text='Correlation Direction',
    yaxis_title='Audio Feature',
    xaxis_title='Correlation with Popularity'
)
st.plotly_chart(fig4)

# Chart 5
st.subheader("Popularity Distribution")
fig5 = px.histogram(filtered_df, 
                    x = 'popularity', 
                    title = 'Popularity Distribution', 
                    nbins = 50, 
                    color_discrete_sequence = ['steelblue'])
fig5.update_layout(
    xaxis_title = 'Popularity Score', 
    yaxis_title = 'Number of Tracks'
)

st.plotly_chart(fig5)