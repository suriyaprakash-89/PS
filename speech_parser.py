import pandas as pd
import speech_recognition as sr

def transcribe_audio(input_file, output_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(input_file) as source:
            print(f"Processing file: {input_file}")
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

            # Save to solution file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ Transcription saved to {output_file}")
            return text
    except Exception as e:
        print(f"❌ Error with {input_file}: {e}")
        return None

if __name__ == "__main__":
    # Load Excel file
    df = pd.read_excel("tasks.xlsx")  # replace with your actual Excel filename

    for index, row in df.iterrows():
        input_file = row["input_file"]
        solution_file = row["solution_file"]

        if pd.notna(input_file) and pd.notna(solution_file):
            transcribe_audio(input_file, solution_file)
