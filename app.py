import streamlit as st
import tensorflow as tf
from music21 import stream, note, chord, text
import numpy as np
import pandas as pd
import kagglehub
import pyphen
import io
import os
import re

st.title("Bach-style Chorale Generator with Lyrics")
st.write("This application generates a four-part Bach-style chorale and aligns it with provided lyrics.")

# Define constants used in data processing
min_note, max_note = 36, 81

# Function to sample the next note based on probabilities
def sample_next_note(probs):
    probabilities = np.asarray(probs, dtype=float)
    prob_sum = probabilities.sum()

    # Fallback to choosing the most probable note if probabilities are invalid
    if prob_sum <= 0 or not np.isfinite(prob_sum):
        return int(np.argmax(probabilities))

    probabilities /= prob_sum

    # Use np.random.choice for sampling
    return np.random.choice(len(probabilities), p=probabilities)

# Function to generate a chorale sequence using the trained model
def generate_chorale(model, seed_chords, length):
    token_sequence = np.array(seed_chords, dtype=int)
    token_sequence = np.where(token_sequence == 0, 0, token_sequence - min_note + 1)
    token_sequence = token_sequence.reshape(1, -1)

    for _ in range(length * 4): # Generate notes for approximately 'length' chords
        # Predict the next note probabilities for all four voices simultaneously
        next_token_probabilities = model.predict(token_sequence)

        # Get the probabilities for the last timestep
        last_timestep_probs = next_token_probabilities[0, -1, :]

        # Sample the next token (note index)
        next_token = sample_next_note(last_timestep_probs)

        # Append the sampled token to the sequence
        token_sequence = np.concatenate([token_sequence, [[next_token]]], axis=1)

    # Convert token sequence back to original pitch values
    token_sequence = np.where(token_sequence == 0, 0, token_sequence + min_note - 1)

    # Reshape to (num_timesteps, 4)
    return token_sequence.reshape(-1, 4)


# Function to align text to the soprano line using syllabification
def align_text_to_soprano_syllables(chorale_text, soprano_line):
    # Initialize the German hyphenator
    dic = pyphen.Pyphen(lang='de_DE')

    # Preprocess the text to remove punctuation and split into words
    processed_text = re.sub(r'[^\w\s]', '', chorale_text)
    processed_text = processed_text.replace('\n', ' ').replace('\r', '')
    words = processed_text.split()

    # Syllabify each word and flatten the list of syllables
    syllables = []
    for word in words:
        syllables.extend(dic.inserted(word).split('-'))

    aligned_output = []
    syllable_index = 0

    for note in soprano_line:
        if syllable_index < len(syllables):
            aligned_output.append({'note': note, 'syllable': syllables[syllable_index]})
            syllable_index += 1
        else:
            # Handle case where there are more notes than syllables
            # For simplicity, repeat the last syllable
            aligned_output.append({'note': note, 'syllable': syllables[-1] if syllables else ''})

    # Handle case where there are more syllables than notes (ignore extra syllables for now)

    return aligned_output

# Load the trained model (assuming it's in the same directory)
@st.cache_resource # Cache the model to avoid reloading
def load_model():
    model = tf.keras.models.load_model('bach_chorale_model.h5')
    return model

model = load_model()

# Load some seed data for generation
# In a real app, you might allow users to upload or select seed data
# For this example, we'll use a pre-downloaded test file
@st.cache_data # Cache the seed data
def load_seed_data():
    # This path assumes the dataset was downloaded to the default kagglehub location
    # Adjust if your environment differs
    dataset_path = kagglehub.dataset_download("pranjalsriv/bach-chorales-2")
    test_files = sorted([os.path.join(dataset_path, 'test', f) for f in os.listdir(os.path.join(dataset_path, 'test')) if f.endswith('.csv')])
    # Load the first test file as seed data
    if test_files:
        return pd.read_csv(test_files[0]).values.tolist()
    return None

seed_data = load_seed_data()

# Streamlit UI for text input
st.header("Enter your chorale text:")
user_text = st.text_area("Enter text here (e.g., hymn lyrics):", chorale_text)

# Streamlit UI for generation parameters
st.header("Generation Settings:")
# You might add more controls here, e.g., length of the generated piece
generation_length = st.slider("Number of chords to generate (approximate):", min_value=10, max_value=200, value=len(user_text.split())*4)

if st.button("Generate Chorale"):
    if seed_data is None:
        st.error("Could not load seed data. Please ensure the dataset is accessible.")
    elif not user_text:
        st.warning("Please enter some text to generate the chorale.")
    else:
        st.info("Generating chorale...")

        # Generate the musical notes
        # Use a slice of the seed data as the starting sequence
        seed_sequence = seed_data[:window_size] # Use window_size from previous definitions
        generated_notes_array = generate_chorale(model, seed_sequence, length=generation_length // 4) # Adjust length based on chords

        # Extract the soprano line
        generated_soprano_line = generated_notes_array[:, 0].tolist()

        # Align the text to the soprano line
        aligned_generated_chorale = align_text_to_soprano_syllables(user_text, generated_soprano_line)

        st.header("Generated Chorale:")

        # Create a music21 stream for the chorale
        chorale_stream = stream.Stream()

        # Create streams for each voice (Soprano, Alto, Tenor, Bass)
        soprano_part = stream.Part()
        alto_part = stream.Part()
        tenor_part = stream.Part()
        bass_part = stream.Part()

        soprano_part.id = 'Soprano'
        alto_part.id = 'Alto'
        tenor_part.id = 'Tenor'
        bass_part.id = 'Bass'

        # Add notes and lyrics to the soprano part
        for item in aligned_generated_chorale:
            if item['note'] > 0:
                n = note.Note(item['note'], quarterLength=1)
                n.addLyric(item['syllable'])
                soprano_part.append(n)
            else:
                r = note.Rest(quarterLength=1)
                soprano_part.append(r)

        # Add notes to the other parts (Alto, Tenor, Bass)
        if generated_notes_array.shape[1] >= 4:
            for row in generated_notes_array:
                # Alto
                if row[1] > 0:
                    alto_part.append(note.Note(row[1], quarterLength=1))
                else:
                    alto_part.append(note.Rest(quarterLength=1))

                # Tenor
                if row[2] > 0:
                    tenor_part.append(note.Note(row[2], quarterLength=1))
                else:
                    tenor_part.append(note.Rest(quarterLength=1))

                # Bass
                if row[3] > 0:
                    bass_part.append(note.Note(row[3], quarterLength=1))
                else:
                    bass_part.append(note.Rest(quarterLength=1))

        # Append the parts to the main chorale stream
        chorale_stream.insert(0, soprano_part)
        chorale_stream.insert(0, alto_part)
        chorale_stream.insert(0, tenor_part)
        chorale_stream.insert(0, bass_part)


        # Display the chorale as a MIDI file
        midi_output = io.BytesIO()
        chorale_stream.write('midi', fp=midi_output)
        st.audio(midi_output.getvalue(), format='audio/midi')

        # Optionally, display the music notation (requires external tools and setup)
        # try:
        #     score_png = chorale_stream.write('musicxml.png')
        #     st.image(score_png)
        # except Exception as e:
        #     st.warning(f"Could not generate music notation image: {e}")
