# Program title: Storytelling App

# Import part
import streamlit as st
from transformers import pipeline

# Function part
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# Main part
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🛬")


<g transform="matrix(0.71 0 0 0.71 12 12)" >
<path style="stroke: none; stroke-width: 1; stroke-dasharray: none; stroke-linecap: butt; stroke-dashoffset: 0; stroke-linejoin: miter; stroke-miterlimit: 4; fill: rgb(0,0,0); fill-rule: nonzero; opacity: 1;" transform=" translate(-16, -15)" d="M 20 1 C 19.590495024733087 1.00042956620741 19.190991107235522 1.1265528354346757 18.855469 1.361328099999999 L 18.8125 1.3378906 C 18.398148 1.1014046 17.952615 1.0016487 17.517578 1.015625 C 16.212468 1.057554 15 2.1193773 15 3.5507812 L 15 7 L 9.6953125 7 C 7.1137278 7 5 9.1137283 5 11.695312 L 5 13.146484 C 3.8491574 13.514911 3 14.588149 3 15.853516 L 3 19.146484 C 3 20.412233 3.8491574 21.485193 5 21.853516 L 5 24.304688 C 5 26.886272 7.1137278 29 9.6953125 29 L 22.304688 29 C 24.886272 29 27 26.886272 27 24.304688 L 27 21.853516 C 28.150843 21.485089 29 20.411851 29 19.146484 L 29 15.853516 C 29 14.587767 28.150843 13.514807 27 13.146484 L 27 11.695312 C 27 9.1137283 24.886272 7 22.304688 7 L 17 7 L 17 3.5507812 C 17 3.0833201 17.413722 2.8421628 17.820312 3.0742188 L 18.011719 3.1835938 C 18.10646862435182 4.211416714908596 18.96782047133229 4.9983099333403835 20 5 C 21.104569499661586 5 22 4.1045694996615865 22 3 C 22 1.895430500338413 21.104569499661586 1 20 1 z M 9.6953125 9 L 22.304688 9 C 23.805103 9 25 10.194898 25 11.695312 L 25 24.304688 C 25 25.805103 23.805103 27 22.304688 27 L 9.6953125 27 C 8.1948972 27 7 25.805103 7 24.304688 L 7 11.695312 C 7 10.194897 8.1948972 9 9.6953125 9 z M 10.707031 12.292969 L 9.2929688 13.707031 L 10.585938 15 L 9.2929688 16.292969 L 10.707031 17.707031 L 12 16.414062 L 13.292969 17.707031 L 14.707031 16.292969 L 13.414062 15 L 14.707031 13.707031 L 13.292969 12.292969 L 12 13.585938 L 10.707031 12.292969 z M 20 13 C 18.895430500338414 13 18 13.895430500338414 18 15 C 18 16.104569499661586 18.895430500338414 17 20 17 C 21.104569499661586 17 22 16.104569499661586 22 15 C 22 13.895430500338414 21.104569499661586 13 20 13 z M 12.212891 21.121094 L 9.7675781 22.753906 L 10.878906 24.417969 L 12.212891 23.527344 L 14.103516 24.789062 L 15.994141 23.525391 L 17.884766 24.787109 L 19.78125 23.525391 L 21.121094 24.417969 L 22.230469 22.753906 L 19.78125 21.123047 L 17.886719 22.384766 L 15.994141 21.121094 L 14.103516 22.382812 L 12.212891 21.121094 z" stroke-linecap="round" />
</g>
</svg>")
st.header("Turn Your Image to Audio Story")
uploaded_file = st.file_uploader("Select an Image...")

if uploaded_file is not None:
    # Save file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Stage 1: Image to Text (Using the function)
    st.text('Processing img2text...')
    scenario = img2text(uploaded_file.name)
    st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story (Inline)
    st.text('Generating a story...')
    story_pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    story_results = story_pipe(scenario)
    story = story_results[0]['generated_text']
    st.write(f"**Story:** {story}")

    # Stage 3: Story to Audio (Inline)
    st.text('Generating audio data...')
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story)

    # Play button
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
